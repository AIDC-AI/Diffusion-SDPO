#!/usr/bin/env bash
set -euo pipefail

# --- defaults ---
DEVICE="cuda"
BATCH_SIZE=64
OUTDIR='output/vs_eval'

LABEL_A=""
LABEL_B=""
JSON_A=""
JSON_B=""


usage() {
  echo "Usage: $0 --json_a A.json --json_b B.json --label_a 'A' --label_b 'B'"
  echo "          [--device cuda|cpu] [--batch_size 64] [--outdir DIR]"
  exit 1
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json_a) JSON_A="$2"; shift 2;;
    --json_b) JSON_B="$2"; shift 2;;
    --label_a) LABEL_A="$2"; shift 2;;
    --label_b) LABEL_B="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "$JSON_A" || -z "$JSON_B" || -z "$LABEL_A" || -z "$LABEL_B" ]] && usage

echo "Label_A: $LABEL_A"
echo "Label_B: $LABEL_B"
echo "Json_A: $JSON_A"
echo "Json_B: $JSON_A"

# --- real output subdir: <OUTDIR>/<A>_vs_<B>_<MMDD-HHMMSS> ---
STAMP="$(date +%m%d-%H%M%S)"
RUN_NAME="${LABEL_A}_vs_${LABEL_B}_${STAMP}"
SUBDIR="${OUTDIR}/${RUN_NAME}"
mkdir -p "$SUBDIR"

echo ">>> Results will be saved under: $SUBDIR"

# --- common flags (explicitly pass in labels to ensure consistency with directory names) ---
COMMON_FLAGS=( --json_a "$JSON_A" --json_b "$JSON_B" --device "$DEVICE" --batch_size "$BATCH_SIZE" \
               --label_a "$LABEL_A" --label_b "$LABEL_B")

# --- run all metrics ---
echo ">>> Running PickScore"
PYTHONPATH='.' python utils/pickscore_vs_utils.py   "${COMMON_FLAGS[@]}" --save "$SUBDIR/pair_pickscore.json"

echo ">>> Running HPSv2"
PYTHONPATH='.' python utils/hps_vs_utils.py       "${COMMON_FLAGS[@]}" --save "$SUBDIR/pair_hpsv2.json"

echo ">>> Running Aesthetics"
PYTHONPATH='.' python utils/aes_vs_utils.py  "${COMMON_FLAGS[@]}" --save "$SUBDIR/pair_aesthetics.json"

echo ">>> Running CLIP"
PYTHONPATH='.' python utils/clip_vs_utils.py   "${COMMON_FLAGS[@]}" --save "$SUBDIR/pair_clip.json"

echo ">>> Running ImageReward"
PYTHONPATH='.' python utils/imagereward_vs_utils.py "${COMMON_FLAGS[@]}" --save "$SUBDIR/pair_imagereward.json"

# --- summary (merge or summarize 5 JSON files) ---
PYTHONPATH='.' python utils/summary_vs_scores.py --outdir $SUBDIR