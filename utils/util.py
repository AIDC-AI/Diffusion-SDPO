# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os
import shutil
import torch
import json
from io import BytesIO
from PIL import Image
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import io
import pandas as pd


class T2ITestBench(Dataset):
    def __init__(self, path='', **kwargs):
        print(f"-> load {path}")
        with open(path, 'r') as f:
            self.dataset = json.load(f)
        print(f"-> find {len(self.dataset)} prompts.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        prompt = data['prompt']
        neg_prompt = data.get("neg_prompt", "")
        raw_image_path = data.get("raw_image_path", "")
        metadata = data.get("metadata", dict())
        result = {
            "index": index,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "raw_image_path": raw_image_path,
            "metadata": metadata,
        }
        return result

    def collate_fn(self, examples):
        index = [example["index"] for example in examples]
        prompt = [example["prompt"] for example in examples]
        neg_prompt = [example["neg_prompt"] for example in examples]
        raw_image_path = [example["raw_image_path"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        return {
            "index": index,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "raw_image_path": raw_image_path,
            "metadata": metadata,
        }


class TxtPrompts(Dataset):
    def __init__(self, path='', **kwargs):
        self.dataset = []
        with open(path, 'r') as fp:
            prompts = fp.readlines()
        self.dataset = [prompt.strip() for prompt in prompts]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prompt = self.dataset[index]
        result = {
            "prompt": prompt,
            "index": index,
        }
        return result

    def collate_fn(self, examples):
        prompt = [example["prompt"] for example in examples]
        index = [example["index"] for example in examples]
        return {
            "prompt": prompt,
            "index": index,
        }


class JsonResultData(Dataset):
    def __init__(self, path='', **kwargs):
        print(f"-> load {path}")
        with open(path, 'r') as f:
            self.dataset = json.load(f)
        print(f"-> find {len(self.dataset)} img-txt pairs.")
        img_size = 256
        self.preprocessing = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prompt = self.dataset[index]['prompt']
        gen_image_path = self.dataset[index].get('gen_image_path', '')
        raw_image_path = self.dataset[index].get('raw_image_path', '')
        instance_image = Image.open(gen_image_path).convert('RGB')
        image_tensor = self.preprocessing(instance_image)
        result = {
            "prompt": prompt,
            "index": index,
            "image_tensor": image_tensor,
            "gen_image_path": gen_image_path,
            "raw_image_path": raw_image_path,
        }
        return result

    def update(self, index, key, value):
        self.dataset[index][key] = value

    def to_json(self):
        return self.dataset[:]

    def collate_fn(self, examples):
        prompt = [example["prompt"] for example in examples]
        index = [example["index"] for example in examples]
        image_tensor = torch.stack([example["image_tensor"] for example in examples])
        gen_image_path = [example["gen_image_path"] for example in examples]
        raw_image_path = [example["raw_image_path"] for example in examples]
        return {
            "prompt": prompt,
            "index": index,
            "image_tensor": image_tensor,
            "gen_image_path": gen_image_path,
            "raw_image_path": raw_image_path,
        }

