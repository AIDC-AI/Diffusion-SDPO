# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os 
from clint.textui import progress
from PIL import Image
import torch
import numpy as np

from utils.tokenizer_hps import HFTokenizer 
from utils.util import JsonResultData
import argparse
import json
from tqdm import tqdm



# import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms

from os.path import join

# import clip
from transformers import AutoProcessor, AutoModel


# if you changed the MLP architecture during training, change it also here:
class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Selector():
    
    def __init__(self, device):
        self.device = device

        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        s = torch.load("utils/aesthetics_model/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)

        self.model.to(device)
        self.model.eval()

        clip_model_name = "openai/clip-vit-large-patch14"
        self.model2 = AutoModel.from_pretrained(clip_model_name).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(clip_model_name)
        # self.model2, self.preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

    def score(self, image_paths, prompt_not_used):
        images = [Image.open(image_path) for image_path in image_paths]
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.model2.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
            return self.model(image_embs).cpu().flatten().tolist()

def parse_args():
    parser = argparse.ArgumentParser(description="Test Aes Score")
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
    aes_score = 0.
    total_samples = 0

    global_step = 0
    for i, data in progress_bar:

        prompt = data["prompt"]
        gen_image_path = data['gen_image_path']
        score = model.score(gen_image_path, prompt)
        n_samples = len(prompt)
        
        aes_score += sum(score)
        total_samples += n_samples
        
        progress_bar.set_postfix({
            "Aes score": np.mean(score).item(),
        })


    assert total_samples == len(dataset)
    print('total len of test data : {}'.format(total_samples))
    print(f"Total Aes score: {aes_score / total_samples}")
    print('FinalResults AesScore for {} : {:.4f}'.format(args.json_path.split('/')[-1].split('_')[0], aes_score / total_samples))
    result = dataset.to_json()
    out_file = args.json_path.replace(".json", "_aesscore.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()