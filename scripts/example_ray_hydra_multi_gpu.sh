#!/usr/bin/env bash

NUM_GPUS=8 && NUM_CPUS_PER_GPU=24 && python -m training.hydra \
  launch_on="$RAY_CONFIG_PATH" \
  hydra.launcher.ray.remote.num_cpus="$(bc <<< $NUM_GPUS*$NUM_CPUS_PER_GPU)" \
  hydra.launcher.ray.remote.num_gpus="$NUM_GPUS" \
  report_to='[tensorboard]' \
  train_data=laion-coco \
  train_num_samples=10_000_000 \
  dataset_resampled=True \
  warmup=2000 \
  batch_size=256 \
  lr=1e-6 \
  wd=0.1 \
  epochs=100 \
  workers="$NUM_CPUS_PER_GPU" \
  model=ViT_B_32 \
  pretrained=openai \
  replace_with_extra_caption=True \
  add_random_text_hard_negatives replace
