#!/usr/bin/env bash

# We are fine with the word splitting here, since we are using printf to escape the spaces.
# shellcheck disable=SC2046
NUM_GPUS=1 && NUM_CPUS_PER_GPU=12 && ray job submit \
  --address "$RAY_ADDRESS" \
  --runtime-env <(./scripts/add_sensitive_env_vars_to_yaml.py conf/hydra/launcher/ray/init/runtime_env/main.yaml) \
  --entrypoint-num-gpus "$NUM_GPUS" \
  --entrypoint-num-cpus "$(bc <<< $NUM_GPUS*$NUM_CPUS_PER_GPU)" \
  --no-wait \
  -- python -m training \
    --report-to tensorboard \
    --train-data laion \
    --warmup 2000 \
    --batch-size 1024 \
    --lr 1e-4 \
    --wd 0.1 \
    --epochs 32 \
    --workers "$NUM_CPUS_PER_GPU" \
    --model ViT-B-32 \
    --pretrained-image \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --resume "$LOG_DIR/name/checkpoints/epoch_9.pt"
