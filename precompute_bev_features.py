"""
Offline BEV feature precomputation for training.

Runs BEV encoder (frozen backbone + seg/det heads) on all LiDAR samples,
saves seg_probs [6,180,180] + det_probs [1,180,180] as fp16 .npz files.

Design principles:
  - BEV encoder only (no LLM) → small memory footprint per GPU
  - Crash-safe: skip already-saved files; on CUDA crash just restart the script
  - 8 GPUs in parallel: each process handles indices where idx % num_gpus == gpu_id

Usage (run from ./):
  # Launch 8 parallel processes in background
  bash scripts/precompute_bev_features.sh
  # Or a single GPU for testing:
  python precompute_bev_features.py --gpu_id 0 --num_gpus 8
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fudoki.janus.models.bev_encoder import BEVFusionEncoder
from flow_matching.data.navsim import load_pcd_points


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_list',    default='data/navsim_103k_bev.jsonl')
    p.add_argument('--output_dir',   default='data/bev_cache')
    p.add_argument('--bev_ckpt',     default='pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')
    p.add_argument('--seg_det_ckpt', default='output/pretrain/seg_det_v3/checkpoint-20000/model.safetensors')
    p.add_argument('--gpu_id',       type=int, default=0)
    p.add_argument('--num_gpus',     type=int, default=8)
    return p.parse_args()


def load_bev_encoder(bev_ckpt: str, seg_det_ckpt: str, device: torch.device) -> BEVFusionEncoder:
    encoder = BEVFusionEncoder(ckpt_path=bev_ckpt, freeze_backbone=True).to(device)

    if seg_det_ckpt and os.path.exists(seg_det_ckpt):
        from safetensors.torch import load_file as st_load_file
        sd = st_load_file(seg_det_ckpt, device='cpu')
        seg_det_sd = {
            k[len('bev_encoder.'):]: v for k, v in sd.items()
            if k.startswith('bev_encoder.seg_head') or k.startswith('bev_encoder.det_head')
        }
        encoder.load_state_dict(seg_det_sd, strict=False)
        print(f"[GPU {device}] Loaded seg/det weights ({len(seg_det_sd)} keys)")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def process_sample(encoder, lidar_path: str, device: torch.device):
    """
    Returns (seg_probs, det_probs) as fp16 numpy arrays, or (None, None) on failure.
    seg_probs: [6, 180, 180] fp16
    det_probs: [1, 180, 180] fp16
    """
    try:
        points = load_pcd_points(lidar_path)   # [N, 4] float32
        if points.shape[0] == 0:
            return None, None

        bev_points = [points]   # bev_encoder expects list of arrays

        with torch.no_grad():
            out = encoder(bev_points, max_voxels=25000)

        if out is None:
            # High voxel count → skipped by threshold guard in bev_encoder
            return None, None

        seg_probs = F.softmax(out['seg_logits'].float(), dim=1)[0]   # [6, 180, 180]
        det_probs = torch.sigmoid(out['det_output']['heatmap'].float())[0]  # [1, 180, 180]

        return seg_probs.half().cpu().numpy(), det_probs.half().cpu().numpy()

    except Exception as e:
        # Catches spconv exceptions that didn't corrupt CUDA context
        warnings.warn(f"[precompute] BEV failed: {e}")
        return None, None


def main():
    args = parse_args()
    # CUDA_VISIBLE_DEVICES is set by the launcher to the correct physical GPU.
    # Inside this process, that GPU is always visible as cuda:0.
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    os.makedirs(args.output_dir, exist_ok=True)
    failed_log = os.path.join(args.output_dir, f'failed_gpu{args.gpu_id}.txt')

    print(f"[GPU {args.gpu_id}] Loading BEV encoder on cuda:0 (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})...")
    encoder = load_bev_encoder(args.bev_ckpt, args.seg_det_ckpt, device)

    # Load data list
    samples = []
    with open(args.data_list, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # This process handles indices where idx % num_gpus == gpu_id
    my_indices = [i for i in range(len(samples)) if i % args.num_gpus == args.gpu_id]
    print(f"[GPU {args.gpu_id}] Handling {len(my_indices)} / {len(samples)} samples")

    # Load previously failed tokens (accumulated across restarts)
    failed_set = set()
    if os.path.exists(failed_log):
        with open(failed_log, 'r') as f:
            failed_set = set(line.strip() for line in f if line.strip())

    # On restart: any leftover .attempt files = crashed mid-sample last run → mark failed
    for fname in os.listdir(args.output_dir):
        if fname.endswith(f'.attempt.gpu{args.gpu_id}'):
            crashed_token = fname.replace(f'.attempt.gpu{args.gpu_id}', '')
            failed_set.add(crashed_token)
            os.remove(os.path.join(args.output_dir, fname))
            print(f"[GPU {args.gpu_id}] Skipping previously crashed token: {crashed_token}")

    if failed_set:
        # Persist updated failed list
        with open(failed_log, 'w') as f:
            f.write('\n'.join(sorted(failed_set)))

    done = skipped = failed = len(failed_set)

    for rank, idx in enumerate(my_indices):
        sample = samples[idx]
        token_id   = sample.get('id', str(idx))
        lidar_path = sample.get('lidar', None)

        save_path   = os.path.join(args.output_dir, f'{token_id}.npz')
        attempt_f   = os.path.join(args.output_dir, f'{token_id}.attempt.gpu{args.gpu_id}')

        # Resume: skip already-computed or permanently failed
        if os.path.exists(save_path):
            skipped += 1
            continue
        if token_id in failed_set:
            continue

        if lidar_path is None or not os.path.exists(lidar_path):
            failed_set.add(token_id)
            with open(failed_log, 'a') as f:
                f.write(token_id + '\n')
            failed += 1
            continue

        # Mark in-progress BEFORE calling spconv (survives a hard crash)
        open(attempt_f, 'w').close()

        seg_probs, det_probs = process_sample(encoder, lidar_path, device)

        if seg_probs is None:
            # Soft failure (caught exception, CUDA context still ok)
            os.remove(attempt_f)
            failed_set.add(token_id)
            with open(failed_log, 'a') as f:
                f.write(token_id + '\n')
            failed += 1
        else:
            np.savez_compressed(save_path, seg_probs=seg_probs, det_probs=det_probs)
            os.remove(attempt_f)
            done += 1

        if rank % 200 == 0:
            print(f"[GPU {args.gpu_id}] {rank}/{len(my_indices)} done={done} skip={skipped} fail={failed}")

    print(f"[GPU {args.gpu_id}] DONE: done={done} skip={skipped} fail={failed}")


if __name__ == '__main__':
    main()
