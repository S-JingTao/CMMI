#!/bin/bash
# Phase 1: pretrain seg_head + det_head only with GT supervision
# Only 2 small heads trainable (~few M params), LLM/backbone all frozen
# Output: output/pretrain/seg_det/



config=config/sft_navsim_seg_det.yaml
output_dir=output/pretrain/seg_det

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
accelerate launch \
  --config_file ./config/accelerate_config_ds2.yaml \
  --machine_rank 0 \
  --main_process_port 12348 \
  --num_machines 1 \
  --num_processes 8 \
  train.py \
  --config $config \
  --output_dir $output_dir
