# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', required=True, type=str)
args = parser.parse_args()

outdir = args.outdir
files = {
  "PickScore":   "pair_pickscore.json",
  "HPSv2":       "pair_hpsv2.json",
  "CLIP":        "pair_clip.json",
  "Aesthetics":  "pair_aesthetics.json",
  "ImageReward": "pair_imagereward.json",
}
rows = []
pair_info = None
for name, fname in files.items():
    path = os.path.join(outdir, fname)
    with open(path, "r") as f:
        d = json.load(f)
    if pair_info is None:
        pair_info = {"model_a": d["model_a"], "model_b": d["model_b"], "ties_policy": d.get("ties_policy","")}
    rows.append({
        "metric": name,
        "mean_a": d["mean_score_a"],
        "mean_b": d["mean_score_b"],
        "win_rate_a_over_b": d["win_rate_a_over_b"],
        "wins_a": d["wins_a"],
        "wins_b": d["wins_b"],
        "ties": d["ties"],
        "total_pairs": d["total_pairs"],
    })

# ---- averages over the 5 reward models ----
assert len(rows) == 5
macro_wr = sum(r["win_rate_a_over_b"] for r in rows) / len(rows)  # unweighted average
total_pairs_all = sum(r["total_pairs"] for r in rows)
micro_wr = (sum(r["win_rate_a_over_b"] * r["total_pairs"] for r in rows) / total_pairs_all) if total_pairs_all > 0 else None

print("\n========== SUMMARY ==========")
print(f"Pair: {pair_info['model_a']}  vs  {pair_info['model_b']}   (ties policy: {pair_info['ties_policy']})\n")
print("| Metric       |  A mean  |  B mean  |  A win rate  | Wins/Ties/Losses |   n  |")
print("|--------------|----------|----------|--------------|------------------|------|")
for r in rows:
    losses_a = r["total_pairs"] - r["wins_a"] - r["ties"]
    wr = 100.0 * r["win_rate_a_over_b"]
    print(f"| {r['metric']:<12} | {r['mean_a']:.6f} | {r['mean_b']:.6f} | {wr:>6.2f}%     | {r['wins_a']}/{r['ties']}/{losses_a}         | {r['total_pairs']:>4} |")

# append average win rate below the table
print("|--------------|----------|----------|--------------|------------------|------|")
print(f"| {'Average':<12} | {'-':>8} | {'-':>8} | {100.0*macro_wr:>6.2f}%     | {'-':<18} | {'-':>4} |")
if micro_wr is not None:
    print(f"\nMacro avg win rate (unweighted across 5 metrics): {100.0*macro_wr:.2f}%")
    print(f"Micro avg win rate (weighted by #pairs):         {100.0*micro_wr:.2f}%")
else:
    print(f"\nMacro avg win rate (unweighted across 5 metrics): {100.0*macro_wr:.2f}%")
print("================================\n")

# write combined json with averages
summary = {
    "pair": pair_info,
    "metrics": rows,
    "averages": {
        "macro_win_rate_a_over_b": macro_wr,   # 0~1
        "micro_win_rate_a_over_b": micro_wr    # range 0–1, may be null
    }
}
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary JSON -> {os.path.join(outdir, 'summary.json')}")