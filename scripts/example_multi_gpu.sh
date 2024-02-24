#!/usr/bin/env bash

OMP_NUM_THREADS=24 torchrun --nproc_per_node 8 -m training \
  --train-data laion-coco \
  --train-num-samples 10_000_000 \
  --dataset-resampled \
  --warmup 2000 \
  --batch-size 256 \
  --lr 1e-6 \
  --wd 0.1 \
  --epochs 100 \
  --workers 24 \
  --model ViT-B-32 \
  --pretrained openai \
  --replace-with-extra-caption \
  --add-random-text-hard-negatives replace
