#!/bin/bash

#evaluation configuration
export SIZE=1024
export STEP=50
export CFG=7.5
export SEED=0


export TEST_JSON=()

MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "Model Path: $0 <model_path>"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "FileNotFoundError: $MODEL_PATH"
  exit 1
fi

export OUTPUT="$(dirname "$MODEL_PATH")"


export TEST_JSON+=("prompts/papv2.json")
export TEST_JSON+=("prompts/hpsv2.json")
export TEST_JSON+=("prompts/partiprompts.json")


export LOG_PATH="${OUTPUT}/joint_test_${SIZE}.log"

exec > >(tee ${LOG_PATH}) 2>&1

get_free_port() {
  local port
  while :; do
    port=$(( ( RANDOM << 15 | RANDOM ) % 50001 + 10000 ))
    if ! (echo >/dev/tcp/127.0.0.1/$port) &>/dev/null; then
      echo "$port"
      return 0
    fi
  done
}

for test_json in "${TEST_JSON[@]}"
do
    echo $test_json

    if [[ "${test_json}" == *papv2* ]]; then
      export DATA="${OUTPUT}/papv2_seed${SEED}_${SIZE}x${SIZE}_${STEP}s_${CFG}cfg.json"
    elif [[ "${test_json}" == *hpsv2* ]]; then
      export DATA="${OUTPUT}/hpsv2_seed${SEED}_${SIZE}x${SIZE}_${STEP}s_${CFG}cfg.json"
    elif [[ "${test_json}" == *partiprompts* ]]; then
      export DATA="${OUTPUT}/partiprompts_seed${SEED}_${SIZE}x${SIZE}_${STEP}s_${CFG}cfg.json"
    fi

    if [[ ! -e "$DATA" ]]; then
      cmd=(python test_sdxl.py \
          --test_json $test_json \
          --unet_path ${MODEL_PATH} \
          --output ${OUTPUT} \
          --height ${SIZE}  \
          --width ${SIZE}  \
          --steps ${STEP} \
          --seed ${SEED} \
          --cfg ${CFG}
      )
      echo "${cmd[@]}"
      "${cmd[@]}"
    fi

    PYTHONPATH='.' python utils/pickscore_utils.py --json_path $DATA 
    PYTHONPATH='.' python utils/hps_utils.py --json_path $DATA 
    PYTHONPATH='.' python utils/aes_utils.py --json_path $DATA
    PYTHONPATH='.' python utils/clip_utils.py --json_path $DATA
    PYTHONPATH='.' python utils/imagereward_utils.py --json_path $DATA


done
PYTHONPATH='.' python utils/gather_log.py --log_path ${LOG_PATH}

