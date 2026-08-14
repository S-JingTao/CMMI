set -x

TRAIN_TEST_SPLIT=navtest

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1

export RAY_LOGGING_LEVEL=ERROR
export RAY_DISABLE_METRICS=1

export CUDA_VISIBLE_DEVICES=2,3


export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
export METRIC_CACHE_PATH="/path/to/metric_cache"

export NAVSIM_EXP_ROOT="exp"


torchrun \
    --nproc_per_node=2 \
    navsim/planning/script/run_pdm_score_cmmi.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=cmmi_agent_eval \
    metric_cache_path=$METRIC_CACHE_PATH \
    agent=cmmi_agent \
    agent.fudoki_path="pretrained_model/fudoki" \
    agent.cmmi_path="pretrained_model/cmmi" \
    agent.text_embedding_path="pretrained_model/fudoki/text_embedding.pt" \
    agent.image_embedding_path="pretrained_model/fudoki/image_embedding.pt" \
    agent.heading_mlp_path="pretrained_model/cmmi/best_model_epoch95.pt" \
    agent.discrete_fm_steps=2