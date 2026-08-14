#!/bin/bash
# Full 40K-step training from official pretrained checkpoint
# L_plan (K=4, div=0.1) + Γ_risk + Γ_smooth, NO BEV token injection into LLM
# Output: output/train/navsim_bev_lplan5_40k/




export PYTHONUNBUFFERED=1

config=config/sft_navsim_bev_lplan5_40k.yaml
output_dir=output/train/navsim_bev_lplan5_40k

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
accelerate launch \
  --config_file ./config/accelerate_config_ds2_zero2.yaml \
  --machine_rank 0 \
  --main_process_port 12348 \
  --num_machines 1 \
  --num_processes 8 \
  train.py \
  --config $config \
  --output_dir $output_dir
