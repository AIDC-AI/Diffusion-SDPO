# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os

import argparse
import re
import time
import json
import zipfile
from dataclasses import dataclass
from glob import iglob
import torch.nn as nn
import math

import torch
from einops import rearrange
from PIL import ExifTags, Image
from torch import Tensor
from tqdm import tqdm
from diffusers.training_utils import EMAModel

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel

import json
from io import BytesIO
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from utils.util import T2ITestBench

def get_noise(
    num_samples: int,
    channel: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        channel,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Test Yak")
    parser.add_argument(
        "--test_json",
        type=str,
        default="prompts/papv2.json",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--steps", type=int, default=50,
    )
    parser.add_argument(
        "--cfg", type=float, default=7.5,
    )
    parser.add_argument(
        "--pretrain_path", type=str, default='stabilityai/stable-diffusion-xl-base-1.0',
    )
    parser.add_argument(
        "--vae_path", type=str, default='madebyollin/sdxl-vae-fp16-fix',
    )

    args = parser.parse_args()
    return args


def load_all_model(model_path, args, dtype=torch.bfloat16):

    pretrain_path = args.pretrain_path
    
    text_encoder_one = CLIPTextModel.from_pretrained(pretrain_path, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrain_path, subfolder="text_encoder_2", torch_dtype=dtype)        
    scheduler = EulerDiscreteScheduler.from_pretrained(pretrain_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrain_path, subfolder="tokenizer_2")

    vae_path = args.vae_path
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder=None, torch_dtype=torch.float32)

    print("Init SDXL model")
    
    if os.path.isdir(model_path):
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrain_path, subfolder="unet", torch_dtype=dtype)
        sd = torch.load(model_path, map_location='cpu')
        msg = unet.load_state_dict(sd)
        print(msg)

    device = 'cuda'

    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    unet.to(device)

    vae.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()
    unet.eval()

    
    return text_encoder_one, text_encoder_two, scheduler, tokenizer, tokenizer_2, vae, unet

def encode_text(tokenizers, text_encoders, prompt):
    prompt_embeds_list = []

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)    
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to('cuda'), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    prompt_embeds = prompt_embeds.to(dtype=text_encoders[-1].dtype, device=text_encoders[-1].device)

    return prompt_embeds, pooled_prompt_embeds

def pipe(components, prompts, height, width, num_steps, seed, cfg=1.0, device='cuda'):

    text_encoder_one, text_encoder_two, scheduler, tokenizer, tokenizer_2, vae, unet = components

    tokenizers = [tokenizer, tokenizer_2]
    text_encoders = [text_encoder_one, text_encoder_two]

    prompt_embeds, pooled_prompt_embeds = encode_text(tokenizers, text_encoders, prompts)
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_text(tokenizers, text_encoders, "")

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    bs = len(prompts)
    channel = vae.config.latent_channels
    height = 16 * (height // 16)
    width = 16 * (width // 16)
    torch_device = torch.device("cuda")

    # prepare input
    latents = get_noise(
        bs,
        channel,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    latents = latents * scheduler.init_noise_sigma

    add_time_ids = torch.tensor([height, width, 0, 0, height, width], dtype=latents.dtype, device=device)[None, :].repeat(latents.size(0), 1)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = scheduler.scale_model_input(latents, t)
            latent_model_input = latent_model_input.repeat(2, 1, 1, 1)

            added_cond_kwargs = {"text_embeds": torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds.repeat(latents.size(0), 1)]), "time_ids": add_time_ids.repeat(2, 1)}

            pred = unet(latent_model_input, t, encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds.repeat(latents.size(0), 1, 1)]), added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]

            pred_win, pred_lose = pred.chunk(2)
            noise_pred = pred_win + cfg * (pred_win - pred_lose)
            latents = scheduler.step(noise_pred, t, latents, generator=None, return_dict=False)[0]

    x = latents.float()

    with torch.no_grad():
        with torch.autocast(device_type=torch_device.type, dtype=torch.float32):
            if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor is not None:
                x = x / vae.config.scaling_factor
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                x = x + vae.config.shift_factor
            x = vae.decode(x, return_dict=False)[0]

    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (x * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def zip_dir(dirname, zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))
        
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        zf.write(tar,arcname)
    zf.close()


def test_t2i_model(dataloader, components, args):
    output_json = []
    for i, data in tqdm(enumerate(dataloader), total= len(dataloader), desc="Testing T2I"):
        prompts = data['prompt']
        indexs = data['index']
        metadatas = data.get('metadata', [{}] * len(prompts))
        raw_image_paths = data.get('raw_image_path', [''] * len(prompts))
        images = pipe(components, prompts, height=args.height, width=args.width, num_steps=args.steps, seed=args.seed, cfg=args.cfg)
        for image, prompt, index, metadata, raw_image_path in zip(images, prompts, indexs, metadatas, raw_image_paths):
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.ImageDescription] = prompt
            gen_img_path = os.path.join(args.output_dir, f"{index}_{args.height}x{args.width}.png")
            image.save(gen_img_path, exif=exif_data)
            output_json.append({
                "prompt": prompt,
                "gen_image_path": gen_img_path,
                "raw_image_path": raw_image_path,
                "metadata": metadata,
            })
    return output_json


def main():
    args = parse_args()
    print(args)

    height = args.height
    width = args.width


    components = load_all_model(args.unet_path, args)

    if args.test_json:
        dataset = T2ITestBench(args.test_json)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    output_dir = os.path.basename(args.test_json).split('.')[0]
    output_dir += "_seed{}".format(args.seed)
    output_dir += f"_{height}x{width}_{args.steps}s_{args.cfg}cfg"
    if args.output_path is None:
        if os.path.isdir(args.unet_path):
          args.output_path = args.unet_path
        else:
          args.output_path = os.path.dirname(args.unet_path)

    output_dir = os.path.join(args.output_path, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    output_json = None
    print(f"output dir: {output_dir}")

    output_json = test_t2i_model(dataloader, components, args)

    zip_dir(output_dir, output_dir + '.zip')
    if output_json is not None:
        with open(output_dir + '.json', 'w') as f:
            json.dump(output_json, f, indent=2)



if __name__ == "__main__":
    main()

