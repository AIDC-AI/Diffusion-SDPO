# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os
import argparse
import json
import numpy as np
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import torchmetrics
from PIL import ExifTags, Image

import ImageReward as RM
from utils.util import JsonResultData


def parse_args():
    parser = argparse.ArgumentParser(description="Test ImageReward Score")
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    #####
    #Currently only batch size 1 is supported!
    #####

    benchmark_type = "ImageReward-v1.0"
    print(f"Loading {benchmark_type} model...")

    model = RM.load(name=benchmark_type, device="cuda")

    print(f"{benchmark_type} benchmark begins!")

    progress_bar = tqdm(enumerate(dataloader), total= len(dataloader), desc="Testing")
    image_reward_score = 0.
    total_samples = 0

    global_step = 0
    for i, data in progress_bar:
        prompt = data["prompt"][0]
        index = data["index"][0]
        image_paths = data["gen_image_path"][0]
        
        with torch.no_grad():
            score = model.score(prompt, image_paths)

        n_samples = 1
        image_reward_score += score

        total_samples += n_samples
        
        dataset.update(index, "image_reward_score", score)
        progress_bar.set_postfix({"ImageReward score": score})


    print(args.json_path)
    print(f"Total ImageReward score: {image_reward_score / total_samples}  over {total_samples} samples")
    print('FinalResults ImageReward for {} : {:.4f}'.format(args.json_path.split('/')[-1].split('_')[0], image_reward_score / total_samples))
    result = dataset.to_json()
    out_file = args.json_path.replace(".json", "_image_reward_score.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)



if __name__ == "__main__":
    main()

