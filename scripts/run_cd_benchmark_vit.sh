#!/bin/bash
datasets=(fgvc caltech101 stanford_cars dtd eurosat oxford_flowers food101 oxford_pets sun397 ucf101)

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python dis_cttta_runner_cd.py \
    --datasets "$dataset" \
    --backbone "ViT-B/16" \
    --output "output/vitb16/cd/" \
    --config "configs/vitb16/${dataset}.xml"
done