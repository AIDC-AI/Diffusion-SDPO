# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os 
import requests
from PIL import Image
import torch
import argparse
import json

from utils.tokenizer_hps import HFTokenizer 
from open_clip import create_model_and_transforms, get_tokenizer
from utils.util import JsonResultData
from tqdm import tqdm


# device = 'cuda'
HF_HUB_PREFIX = 'hf-hub:'


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.model, preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True
        )


        self.tokenizer = get_tokenizer('ViT-H-14')

    @torch.no_grad
    def score(self, img_path, prompt):
        assert isinstance(img_path, list)
        image_tensor = []
        for one_img_path in img_path:
            if isinstance(one_img_path, str):
                image = self.preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
            elif isinstance(one_img_path, Image.Image):
                image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
            else:
                raise TypeError('The type of parameter img_path is illegal.')
            image_tensor.append(image)
        image_tensor = torch.cat(image_tensor, dim=0)

        text = self.tokenizer(prompt).to(device=self.device, non_blocking=True)

        outputs = self.model(image_tensor, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T

        clip_score = torch.diagonal(logits_per_image).cpu().numpy().reshape(-1)

        return clip_score


def parse_args():
    parser = argparse.ArgumentParser(description="Test Clip Score")
    parser.add_argument(
        "--json_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataset = JsonResultData(args.json_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    device='cuda'
    model = Selector(device)

    progress_bar = tqdm(enumerate(dataloader), total= len(dataloader), desc="Testing")
    clip_score = 0.
    total_samples = 0

    global_step = 0
    for i, data in progress_bar:

        prompt = data["prompt"]
        gen_image_path = data['gen_image_path']
        score = model.score(gen_image_path, prompt)
        n_samples = len(prompt)
        
        clip_score += score.sum()
        total_samples += n_samples
        
        progress_bar.set_postfix({
            "CLIP score": score.mean().item(),
        })


    assert total_samples == len(dataset)
    print('total len of test data : {}'.format(total_samples))
    print(f"Total CLIP score: {clip_score / total_samples}")
    print('FinalResults CLIPScore for {} : {:.4f}'.format(args.json_path.split('/')[-1].split('_')[0], clip_score / total_samples))
    result = dataset.to_json()
    out_file = args.json_path.replace(".json", "_clipscore.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()

