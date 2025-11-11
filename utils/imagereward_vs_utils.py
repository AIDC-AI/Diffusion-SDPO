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

from tqdm import tqdm
import ImageReward as RM


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
            # strict check to avoid mismatches inside bucket
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
    parser = argparse.ArgumentParser(description="Pairwise ImageReward win-rate evaluation")
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

    # Load ImageReward model (single-sample API)
    benchmark_type = "ImageReward-v1.0"
    print(f"Loading {benchmark_type} model on {args.device} ...")
    model = RM.load(name=benchmark_type, device=args.device)
    print("Model loaded.")

    scores_a, scores_b = [], []
    wins_a = wins_b = ties = 0

    for p, pa, pb in tqdm(zip(prompts, imgs_a, imgs_b), total=n, desc="Scoring (ImageReward)"):
        # Each call scores a single (prompt, image_path)
        sa = float(model.score(p, pa))
        sb = float(model.score(p, pb))
        scores_a.append(sa)
        scores_b.append(sb)
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
    print(f"Mean ImageReward -> {label_a}: {mean_a:.6f} | {label_b}: {mean_b:.6f}")
    print(f"Wins -> {label_a}: {wins_a} | {label_b}: {wins_b} | ties: {ties}")
    if args.ties_as_loss_for_a:
        print(f"Win rate {label_a} over {label_b} (ties=0): {100.0 * win_rate_a:.2f}%")
    else:
        print(f"Win rate {label_a} over {label_b} (ties=0.5): {100.0 * win_rate_a:.2f}%")
    print("=" * 60)

    if args.save is None:
        base_a = os.path.splitext(os.path.basename(args.json_a))[0]
        base_b = os.path.splitext(os.path.basename(args.json_b))[0]
        args.save = f"{base_a}_vs_{base_b}_pickscore_results.json"

    # Per-prompt results
    results = []
    for i, (p, pa, pb, sa, sb) in enumerate(zip(prompts, imgs_a, imgs_b, scores_a, scores_b)):
        winner = label_a if sa > sb else (label_b if sb > sa else "tie")
        results.append({
            "pair_index": i,
            "prompt": p,
            "image_a": pa,
            "image_b": pb,
            "score_a": float(sa),
            "score_b": float(sb),
            "winner": winner
        })

    # JSON schema kept identical for easy aggregation
    summary = {
        "model_a": label_a,
        "model_b": label_b,
        "total_pairs": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a_over_b": win_rate_a,  # fraction in [0,1]
        "mean_score_a": mean_a,       # field name kept for compatibility (actually ImageReward mean)
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
