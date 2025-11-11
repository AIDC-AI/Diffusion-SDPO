# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os 
import requests
from clint.textui import progress
from PIL import Image
import torch

from utils.tokenizer_hps import HFTokenizer 
from open_clip import create_model_and_transforms, get_tokenizer
from utils.util import JsonResultData
import argparse
import json
from tqdm import tqdm

root_path = "models-hpsv2"
HF_HUB_PREFIX = 'hf-hub:'
# def get_tokenizer(model_name):
#     if model_name.startswith(HF_HUB_PREFIX):
#         tokenizer = HFTokenizer(model_name[len(HF_HUB_PREFIX):])
#     else:
#         config = get_model_config(model_name)
#         tokenizer = HFTokenizer(
#             config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
#     return tokenizer


class Selector():
    
    def __init__(self, device):
        self.device = device 
        model, preprocess_train, self.preprocess_val = create_model_and_transforms(
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

        # check if the default checkpoint exists
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        checkpoint_path = os.path.join(root_path, 'HPS_v2_compressed.pt')
        if not os.path.exists(checkpoint_path):
            print('Downloading HPS_v2_compressed.pt ...')
            url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
            r = requests.get(url, stream=True)
            with open(os.path.join(root_path, 'HPS_v2_compressed.pt'), 'wb') as HPSv2:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print('Download HPS_v2_compressed.pt to {} sucessfully.'.format(root_path+'/'))


        print('Loading model ...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(device)
        model.eval()
        self.model = model
        print('Loading model successfully!')

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

        hps_score = torch.diagonal(logits_per_image).cpu().numpy().reshape(-1)

        return hps_score

def parse_args():
    parser = argparse.ArgumentParser(description="Test HPSv2 Score")
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
    hps_score = 0.
    total_samples = 0

    global_step = 0
    for i, data in progress_bar:

        prompt = data["prompt"]
        gen_image_path = data['gen_image_path']
        score = model.score(gen_image_path, prompt)
        n_samples = len(prompt)
        
        hps_score += score.sum()
        total_samples += n_samples
        
        progress_bar.set_postfix({
            "HPSv2 score": score.mean().item(),
        })


    assert total_samples == len(dataset)
    print('total len of test data : {}'.format(total_samples))
    print(f"Total HPSv2 score: {hps_score / total_samples}")
    print('FinalResults HPSv2 for {} : {:.4f}'.format(args.json_path.split('/')[-1].split('_')[0], hps_score / total_samples))
    result = dataset.to_json()
    out_file = args.json_path.replace(".json", "_hpsv2score.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()

