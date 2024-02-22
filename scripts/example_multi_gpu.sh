#!/usr/bin/env bash

OMP_NUM_THREADS=12 torchrun --nproc_per_node 2 -m training \
  --copy-codebase \
  --remote-sync "$LOG_DIR" \
  --report-to neptune tensorboard \
  --train-data laion \
  --warmup 2000 \
  --batch-size 1024 \
  --lr 1e-4 \
  --wd 0.1 \
  --epochs 32 \
  --workers 12 \
  --model ViT-B-32 \
  --pretrained-image \
  --lock-text \
  --lock-text-freeze-layer-norm \
  --resume "$LOG_DIR/name/checkpoints/epoch_9.pt"
