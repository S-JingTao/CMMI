#!/bin/bash
# Evaluate CMMIHybridAgent:
#   front-end = baseline (no BEV token in LLM)
#   back-end  = K=5 DFM trajectories + GPU LM optimisation with seg/det from BEV encoder

set -x

TRAIN_TEST_SPLIT=navtest

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63671}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
export METRIC_CACHE_PATH="/path/to/metric_cache"
export NAVSIM_EXP_ROOT="exp"



torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    navsim/planning/script/run_pdm_score_cmmi.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=cmmi_hybrid_eval \
    metric_cache_path=$METRIC_CACHE_PATH \
    agent=cmmi_hybrid_agent \
    agent.fudoki_path="pretrained_model/fudoki" \
    agent.cmmi_path="pretrained_model/cmmi" \
    agent.bev_ckpt_path="pretrained_model/bevfusion/bevfusion_lidar_nusc.pth" \
    agent.seg_det_ckpt_path="output/pretrain/seg_det/checkpoint-20000/model.safetensors" \
    agent.text_embedding_path="pretrained_model/fudoki/text_embedding.pt" \
    agent.image_embedding_path="pretrained_model/fudoki/image_embedding.pt" \
    agent.heading_mlp_path="pretrained_model/cmmi/best_model_epoch95.pt" \
    agent.discrete_fm_steps=5
