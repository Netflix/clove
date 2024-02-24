#!/usr/bin/env bash

# We are fine with the word splitting here, since we are using printf to escape the spaces.
# shellcheck disable=SC2046
NUM_GPUS=1 && NUM_CPUS_PER_GPU=24 && ray job submit \
  --address "$RAY_ADDRESS" \
  --runtime-env <(./scripts/add_sensitive_env_vars_to_yaml.py conf/hydra/launcher/ray/init/runtime_env/main.yaml) \
  --entrypoint-num-gpus "$NUM_GPUS" \
  --entrypoint-num-cpus "$(bc <<< $NUM_GPUS*$NUM_CPUS_PER_GPU)" \
  --no-wait \
  -- python -m training \
    --report-to tensorboard \
    --train-data laion-coco \
    --train-num-samples 10_000_000 \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size 256 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 100 \
    --workers "$NUM_CPUS_PER_GPU" \
    --model ViT-B-32 \
    --pretrained openai \
    --replace-with-extra-caption \
    --add-random-text-hard-negatives replace
