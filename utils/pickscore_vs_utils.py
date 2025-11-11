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
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm import tqdm


PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "yuvalkirstain/PickScore_v1"


def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize fields that we rely on
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


class Selector:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).eval().to(self.device)

    @torch.no_grad()
    def score_pairs(self, img_paths_a: List[str], img_paths_b: List[str], prompts: List[str],
                    batch_size: int = 64, softmax: bool = False) -> Tuple[List[float], List[float]]:
        """
        Computes PickScore for (prompt_i, imgA_i) and (prompt_i, imgB_i) in batches.
        Returns two lists of floats: scores_a, scores_b (same length as prompts).
        """
        assert len(img_paths_a) == len(img_paths_b) == len(prompts)
        n = len(prompts)
        scores_a, scores_b = [], []

        for start in tqdm(range(0, n, batch_size), desc="Scoring", leave=False):
            end = min(start + batch_size, n)
            batch_prompts = prompts[start:end]
            batch_imgs_a = img_paths_a[start:end]
            batch_imgs_b = img_paths_b[start:end]

            # Load images
            imgs = [Image.open(p).convert("RGB") for p in (batch_imgs_a + batch_imgs_b)]

            # Prepare inputs: concatenate A and B; duplicate prompts accordingly
            image_inputs = self.processor(
                images=imgs, padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)

            text_inputs = self.processor(
                text=(batch_prompts + batch_prompts),
                padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)

            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            sim = text_embs @ image_embs.T  # [2B, 2B]

            # diagonal gives pair-aligned scores:
            # first B diagonals -> (prompt_i, imgA_i), next B diagonals -> (prompt_i, imgB_i)
            diag = torch.diagonal(sim)
            a = diag[: len(batch_prompts)]
            b = diag[len(batch_prompts):]

            if softmax:
                logit_scale = self.model.logit_scale.exp()
                logits = torch.stack([a, b], dim=1) * logit_scale  # [B,2]
                probs = torch.softmax(logits, dim=1)               # [B,2]
                a = probs[:, 0]
                b = probs[:, 1]

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
        # count leftovers as dropped
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
    parser = argparse.ArgumentParser(description="Pairwise PickScore win-rate evaluation")
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

    label_a = args.label_a or '/'.join(args.json_a.split('/')[-3:])
    label_b = args.label_b or '/'.join(args.json_b.split('/')[-3:])

    prompts, imgs_a, imgs_b, idx_map = align_pairs(data_a, data_b)
    n = len(prompts)
    if n == 0:
        raise ValueError("No pairs to compare. Check your JSON inputs.")

    selector = Selector(device=args.device)
    scores_a, scores_b = selector.score_pairs(imgs_a, imgs_b, prompts, batch_size=args.batch_size, softmax=False)

    # Compute wins
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
        # default: ties as half
        win_rate_a = (wins_a + 0.5 * ties) / n

    win_rate_b = 1.0 - win_rate_a if not args.ties_as_loss_for_a else (wins_b + 0.5 * ties) / n

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    print("=" * 60)
    print(f"Total pairs: {n}")
    print(f"Model A: {label_a}")
    print(f"Model B: {label_b}")
    print(f"Mean PickScore -> {label_a}: {mean_a:.6f} | {label_b}: {mean_b:.6f}")
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
