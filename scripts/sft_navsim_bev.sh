#!/bin/bash



config=config/sft_navsim_bev.yaml
output_dir=output/train/navsim_bev

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
accelerate launch \
  --config_file ./config/accelerate_config_ds2.yaml \
  --machine_rank 0 \
  --main_process_port 12346 \
  --num_machines 1 \
  --num_processes 8 \
  train.py \
  --config $config \
  --output_dir $output_dir
