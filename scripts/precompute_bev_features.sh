#!/bin/bash
# Launch 8 parallel BEV feature precomputation processes (one per GPU).
# Crash-safe: each process handles its own GPU. On CUDA crash, just re-run this
# script — already-saved .npz files are skipped automatically.
#
# Usage:
#   
#   bash scripts/precompute_bev_features.sh
#   # Monitor:
#   tail -f /tmp/bev_precompute_gpu*.log



export PYTHONUNBUFFERED=1

NUM_GPUS=8
DATA_LIST=data/navsim_103k_bev.jsonl
OUTPUT_DIR=data/bev_cache
BEV_CKPT=pretrained_model/bevfusion/bevfusion_lidar_nusc.pth
SEG_DET_CKPT=output/pretrain/seg_det_v3/checkpoint-20000/model.safetensors

mkdir -p $OUTPUT_DIR

# Worker function: run with auto-restart on CUDA crash.
# Each crash is detected via leftover .attempt.gpuN files → crashed token skipped next run.
run_gpu_worker() {
    GPU_ID=$1
    LOG=/tmp/bev_precompute_gpu${GPU_ID}.log
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
    LOG=/tmp/bev_precompute_gpu${GPU_ID}.log
    echo "Launching GPU ${GPU_ID} → ${LOG}"
    run_gpu_worker $GPU_ID &
    echo "  PID: $!"
done

echo "All 8 workers launched with auto-restart."
echo "Monitor:  tail -f /tmp/bev_precompute_gpu*.log"
echo "Progress: ls ./data/bev_cache/*.npz | wc -l"
