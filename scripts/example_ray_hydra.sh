#!/usr/bin/env bash

NUM_GPUS=1 && NUM_CPUS_PER_GPU=12 && python -m training.hydra \
  launch_on="$RAY_CONFIG_PATH" \
  hydra.launcher.ray.remote.num_cpus="$(bc <<< $NUM_GPUS*$NUM_CPUS_PER_GPU)" \
  hydra.launcher.ray.remote.num_gpus="$NUM_GPUS" \
  report_to='[tensorboard]' \
  train_data=laion \
  warmup=2000 \
  batch_size=1024 \
  lr=1e-4 \
  wd=0.1 \
  epochs=32 \
  workers="$NUM_CPUS_PER_GPU" \
  model=ViT-B-32 \
  pretrained_image=True \
  lock_text=True \
  lock_text_freeze_layer_norm=True \
  resume="$LOG_DIR/name/checkpoints/epoch_9.pt"
