#!/bin/bash

NUM_GPUS=$(python - <<'PY'
import torch; print(torch.cuda.device_count())
PY
)

MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE="madebyollin/sdxl-vae-fp16-fix"
DATASET_NAME="yuvalkirstain/pickapic_v2"
OUTPUT_DIR="output/sdxl-dspo"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048.

accelerate launch --num_processes $NUM_GPUS train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=2000 \
  --lr_scheduler='constant_with_warmup' --lr_warmup_steps=500 \
  --learning_rate=1e-9 --scale_lr \
  --checkpointing_steps 500 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --output_dir $OUTPUT_DIR \
  --sdxl \
  --train_method dspo \
  --beta_dpo 5000 \
  --use_winner_preserving \
  --winner_preserving_mu 0.6