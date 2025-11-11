# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

import hpsv2
import requests
from clint.textui import progress

from utils.tokenizer_hps import HFTokenizer
from open_clip import create_model_and_transforms, get_tokenizer

# ---- HPSv2 checkpoint root ----
ROOT_PATH = "models-hpsv2"
CKPT_NAME = "HPS_v2_compressed.pt"
CKPT_URL = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"


def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    normalized = []
    for i, x in enumerate(data):
        prompt = x.get("prompt", None)
        img = x.get("gen_image_path", None)
        # If gen_image_path is a list, take the first non-empty path
        if isinstance(img, list):
            img = next((p for p in img if isinstance(p, str) and len(p) > 0), "")
        if not isinstance(prompt, str) or not isinstance(img, str):
            continue
        normalized.append({"idx": i, "prompt": prompt, "gen_image_path": img})
    return normalized


def group_by_prompt(entries: List[Dict]) -> Dict[str, List[Dict]]:
    bucket = defaultdict(list)
    for e in entries:
        bucket[e["prompt"]].append(e)
    return bucket


def last_k(path: str, k: int = 3) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return "/".join(parts[-k:])


class Selector:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # build model & preprocess
        model, _preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
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

        # ensure checkpoint exists
        os.makedirs(ROOT_PATH, exist_ok=True)
        ckpt_path = os.path.join(ROOT_PATH, CKPT_NAME)
        if not os.path.exists(ckpt_path):
            print(f"Downloading {CKPT_NAME} ...")
            r = requests.get(CKPT_URL, stream=True)
            with open(ckpt_path, 'wb') as f:
                total_length = int(r.headers.get('content-length', 0)) or None
                if total_length:
                    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                else:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
            print(f"Downloaded to {ROOT_PATH}/")

        # load weights
        print("Loading HPSv2 model ...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = model.to(self.device).eval()
        print("Model loaded.")

    @torch.no_grad()
    def score_pairs(self, img_paths_a: List[str], img_paths_b: List[str], prompts: List[str],
                    batch_size: int = 64) -> Tuple[List[float], List[float]]:
        """
        Computes HPSv2 score for (prompt_i, imgA_i) and (prompt_i, imgB_i) in batches.
        Returns: scores_a, scores_b (lists of floats; same length as prompts).
        """
        assert len(img_paths_a) == len(img_paths_b) == len(prompts)
        n = len(prompts)
        scores_a, scores_b = [], []

        for start in tqdm(range(0, n, batch_size), desc="Scoring (HPSv2)", leave=False):
            end = min(start + batch_size, n)
            batch_prompts = prompts[start:end]
            batch_imgs_a = img_paths_a[start:end]
            batch_imgs_b = img_paths_b[start:end]

            # load + preprocess images (A then B)
            img_tensors = []
            for p in (batch_imgs_a + batch_imgs_b):
                if isinstance(p, str):
                    with Image.open(p) as im:
                        img_tensors.append(self.preprocess_val(im).unsqueeze(0))
                elif isinstance(p, Image.Image):
                    img_tensors.append(self.preprocess_val(p).unsqueeze(0))
                else:
                    raise TypeError("img path must be str or PIL.Image")
            image_tensor = torch.cat(img_tensors, dim=0).to(self.device, non_blocking=True)

            # tokenize text: duplicate prompts for A and B
            text = self.tokenizer(batch_prompts + batch_prompts).to(self.device, non_blocking=True)

            outputs = self.model(image_tensor, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T  # [2B, 2B]
            diag = torch.diagonal(logits_per_image)
            a = diag[: len(batch_prompts)]
            b = diag[len(batch_prompts):]

            scores_a.extend(a.detach().cpu().tolist())
            scores_b.extend(b.detach().cpu().tolist())

        return scores_a, scores_b


def align_pairs(
    data_a: List[Dict], data_b: List[Dict]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Align two lists by prompt. Handles duplicates by pairing in index order within each prompt bucket.
    Returns: prompts, img_paths_a, img_paths_b, mapping_indices (indices from original A list for reference).
    """
    by_prompt_a = group_by_prompt(data_a)
    by_prompt_b = group_by_prompt(data_b)

    common_prompts = sorted(set(by_prompt_a.keys()) & set(by_prompt_b.keys()))
    prompts_out, imgs_a, imgs_b, idx_a_list = [], [], [], []

    dropped = 0
    for p in common_prompts:
        list_a = by_prompt_a[p]
        list_b = by_prompt_b[p]
        k = min(len(list_a), len(list_b))
        if k == 0:
            continue
        for i in range(k):
            assert p == list_a[i]["prompt"]
            assert p == list_b[i]["prompt"]
            prompts_out.append(p)
            imgs_a.append(list_a[i]["gen_image_path"])
            imgs_b.append(list_b[i]["gen_image_path"])
            idx_a_list.append(list_a[i]["idx"])
        dropped += abs(len(list_a) - len(list_b))

    if len(prompts_out) == 0:
        # fallback: align by index if nothing matched (e.g., slightly different prompt strings)
        k = min(len(data_a), len(data_b))
        prompts_out = [data_a[i]["prompt"] for i in range(k)]
        imgs_a = [data_a[i]["gen_image_path"] for i in range(k)]
        imgs_b = [data_b[i]["gen_image_path"] for i in range(k)]
        idx_a_list = [data_a[i]["idx"] for i in range(k)]
        print("[WARN] No exact prompt matches. Falling back to index-wise pairing.")
    else:
        if dropped > 0:
            print(f"[INFO] Aligned on {len(prompts_out)} pairs; dropped {dropped} unmatched or extra items due to duplicates.")

    return prompts_out, imgs_a, imgs_b, idx_a_list


def main():
    parser = argparse.ArgumentParser(description="Pairwise HPSv2 win-rate evaluation")
    parser.add_argument("--json_a", type=str, required=True, help="Path to model A json")
    parser.add_argument("--json_b", type=str, required=True, help="Path to model B json")
    parser.add_argument("--label_a", type=str, default=None, help="Readable name for model A")
    parser.add_argument("--label_b", type=str, default=None, help="Readable name for model B")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ties_as_half", action="store_true", help="Count ties as 0.5 (default).")
    parser.add_argument("--ties_as_loss_for_a", action="store_true", help="Count ties as 0 for A (overrides --ties_as_half).")
    parser.add_argument("--save", type=str, default=None, help="Output JSON path with per-prompt results")
    args = parser.parse_args()

    data_a = load_json(args.json_a)
    data_b = load_json(args.json_b)

    label_a = args.label_a or last_k(args.json_a)
    label_b = args.label_b or last_k(args.json_b)

    prompts, imgs_a, imgs_b, idx_map = align_pairs(data_a, data_b)
    n = len(prompts)
    if n == 0:
        raise ValueError("No pairs to compare. Check your JSON inputs.")

    selector = Selector(device=args.device)
    scores_a, scores_b = selector.score_pairs(imgs_a, imgs_b, prompts, batch_size=args.batch_size)

    # Compute wins/ties
    wins_a = 0
    wins_b = 0
    ties = 0
    for sa, sb in zip(scores_a, scores_b):
        if sa > sb:
            wins_a += 1
        elif sb > sa:
            wins_b += 1
        else:
            ties += 1

    if args.ties_as_loss_for_a:
        win_rate_a = wins_a / n
    else:
        win_rate_a = (wins_a + 0.5 * ties) / n  # default ties=0.5

    win_rate_b = 1.0 - win_rate_a if not args.ties_as_loss_for_a else (wins_b + 0.5 * ties) / n

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    print("=" * 60)
    print(f"Total pairs: {n}")
    print(f"Model A: {label_a}")
    print(f"Model B: {label_b}")
    print(f"Mean HPSv2 -> {label_a}: {mean_a:.6f} | {label_b}: {mean_b:.6f}")
    print(f"Wins -> {label_a}: {wins_a} | {label_b}: {wins_b} | ties: {ties}")
    if args.ties_as_loss_for_a:
        print(f"Win rate {label_a} over {label_b} (ties=0): {100.0 * win_rate_a:.2f}%")
    else:
        print(f"Win rate {label_a} over {label_b} (ties=0.5): {100.0 * win_rate_a:.2f}%")
    print("=" * 60)

    # Save per-prompt results
    if args.save is None:
        base_a = os.path.splitext(os.path.basename(args.json_a))[0]
        base_b = os.path.splitext(os.path.basename(args.json_b))[0]
        args.save = f"{base_a}_vs_{base_b}_pickscore_results.json"

    results = []
    for i, (p, pa, pb, sa, sb) in enumerate(zip(prompts, imgs_a, imgs_b, scores_a, scores_b)):
        if sa > sb:
            winner = label_a
        elif sb > sa:
            winner = label_b
        else:
            winner = "tie"
        results.append({
            "pair_index": i,
            "prompt": p,
            "image_a": pa,
            "image_b": pb,
            "score_a": float(sa),
            "score_b": float(sb),
            "winner": winner
        })

    summary = {
        "model_a": label_a,
        "model_b": label_b,
        "total_pairs": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a_over_b": win_rate_a,  # fraction in [0,1]
        "mean_score_a": mean_a,
        "mean_score_b": mean_b,
        "ties_policy": "half" if not args.ties_as_loss_for_a else "loss_for_a",
        "results": results
    }

    with open(args.save, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved detailed results to: {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
