#!/bin/bash
# Precompute BEV features for navtest tokens.
# Output goes into data/bev_cache (same dir as trainval cache).
# Usage:
#   
#   bash scripts/precompute_navtest_bev.sh



export PYTHONUNBUFFERED=1

NUM_GPUS=8
DATA_LIST=data/navtest_bev.jsonl
OUTPUT_DIR=data/bev_cache
BEV_CKPT=pretrained_model/bevfusion/bevfusion_lidar_nusc.pth
SEG_DET_CKPT=output/pretrain/seg_det_v3/checkpoint-20000/model.safetensors

mkdir -p $OUTPUT_DIR

run_gpu_worker() {
    GPU_ID=$1
    LOG=/tmp/bev_navtest_gpu${GPU_ID}.log
    RESTART=0
    while true; do
        if [ $RESTART -gt 0 ]; then
            echo "[GPU $GPU_ID] Restart #$RESTART" >> $LOG
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python precompute_bev_features.py \
            --gpu_id $GPU_ID \
            --num_gpus $NUM_GPUS \
            --data_list $DATA_LIST \
            --output_dir $OUTPUT_DIR \
            --bev_ckpt $BEV_CKPT \
            --seg_det_ckpt $SEG_DET_CKPT \
            >> $LOG 2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[GPU $GPU_ID] Finished successfully." >> $LOG
            break
        else
            echo "[GPU $GPU_ID] Crashed (exit=$EXIT_CODE), restarting in 3s..." >> $LOG
            sleep 3
            RESTART=$((RESTART + 1))
        fi
    done
}

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG=/tmp/bev_navtest_gpu${GPU_ID}.log
    echo "Launching GPU ${GPU_ID} → ${LOG}"
    run_gpu_worker $GPU_ID &
    echo "  PID: $!"
done

echo "All 8 workers launched."
echo "Monitor:  tail -f /tmp/bev_navtest_gpu*.log"
echo "Progress: ls ./data/bev_cache/*.npz | wc -l"
