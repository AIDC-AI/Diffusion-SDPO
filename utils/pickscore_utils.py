# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import argparse

from utils.util import JsonResultData
from tqdm import tqdm
import json

# load model

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    def score(self, image_paths, prompt, softmax=False):
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores =  (text_embs @ image_embs.T)

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                # get probabilities if you have multiple images to choose from
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            else:
                return torch.diagonal(scores).cpu().numpy().reshape(-1)

def parse_args():
    parser = argparse.ArgumentParser(description="Test PickScore")
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
    pick_score = 0.
    total_samples = 0

    global_step = 0
    for i, data in progress_bar:

        prompt = data["prompt"]
        gen_image_path = data['gen_image_path']
        score = model.score(gen_image_path, prompt)
        n_samples = len(prompt)
        
        pick_score += score.sum()
        total_samples += n_samples
        
        progress_bar.set_postfix({
            "Pick score": score.mean().item(),
        })


    assert total_samples == len(dataset)
    print('total len of test data : {}'.format(total_samples))
    print(f"Total Pick score: {pick_score / total_samples}")
    print('FinalResults PickScore for {} : {:.4f}'.format(args.json_path.split('/')[-1].split('_')[0], pick_score / total_samples))
    result = dataset.to_json()
    out_file = args.json_path.replace(".json", "_pickscore.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
