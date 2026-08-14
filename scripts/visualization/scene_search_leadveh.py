"""
搜索：路口 + 前方有车辆 + BEV seg/det 能跑通的场景。

用 GT 场景元数据判断是否有前车（不依赖 BEV 检测），然后验证 spconv 是否成功。

Usage:
  
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate cmmi
  CUDA_VISIBLE_DEVICES=1 python scripts/visualization/scene_search_leadveh.py [--n-scan 300]
"""

import os, sys, argparse, random
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['LOCAL_RANK']            = '0'
os.environ['OPENSCENE_DATA_ROOT']   = '/path/to/dataset'
os.environ['NUPLAN_MAPS_ROOT']      = '/path/to/dataset/maps'

import numpy as np
import torch
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/path/to/dataset')

OUR_CKPT    = str(ROOT / 'output/train/navsim_bev_lplan5/checkpoint-10000')
BEV_CKPT    = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')
FUDOKI      = str(ROOT / 'pretrained_model/fudoki')
TEXT_EMB    = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt')
IMAGE_EMB   = str(ROOT / 'pretrained_model/fudoki/image_embedding.pt')
HEADING_MLP = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt')

MAX_VOXELS  = 20000

# Lead vehicle search window (ego frame)
LEAD_FWD_MIN = 2.0   # at least 2m ahead
LEAD_FWD_MAX = 30.0  # within 30m ahead
LEAD_LAT_MAX = 4.0   # within ±4m lateral (roughly same lane)


def build_agent():
    from navsim.agents.cmmi_bev_agent import CMMIBevAgent
    agent = CMMIBevAgent(
        fudoki_path          = FUDOKI,
        cmmi_path        = OUR_CKPT,
        bev_ckpt_path        = BEV_CKPT,
        text_embedding_path  = TEXT_EMB,
        image_embedding_path = IMAGE_EMB,
        heading_mlp_path     = HEADING_MLP,
        discrete_fm_steps    = 10,
    )
    agent.initialize()
    return agent


def load_scene(token: str):
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='lv'):
        sf = instantiate(hydra.compose(config_name='navtest'))
    sf.tokens = [token]

    loader = SceneLoader(
        sensor_blobs_path=DATA_ROOT / 'sensor_blobs/test',
        data_path=DATA_ROOT / 'navsim_logs/test',
        scene_filter=sf,
        sensor_config=SensorConfig(
            cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[],
            cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
            lidar_pc=[3],
        ),
    )
    return loader.get_scene_from_token(token)


def get_navtest_tokens():
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='all'):
        sf = instantiate(hydra.compose(config_name='navtest'))
    loader = SceneLoader(
        sensor_blobs_path=DATA_ROOT / 'sensor_blobs/test',
        data_path=DATA_ROOT / 'navsim_logs/test',
        scene_filter=sf,
        sensor_config=SensorConfig(
            cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[],
            cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
            lidar_pc=[3],
        ),
    )
    return list(loader.tokens)


def has_lead_vehicle(scene):
    """Check if current frame has a vehicle ahead using GT annotation boxes.

    NavSim box format: [x, y, z, length, width, height, yaw] in ego frame.
    ego frame: x=forward, y=left(lateral), z=up.
    """
    try:
        hist_frames = scene.scene_metadata.num_history_frames
        cur_frame = scene.frames[hist_frames - 1]
        ann = cur_frame.annotations
        if ann is None or not hasattr(ann, 'boxes') or len(ann.boxes) == 0:
            return False, 0, 0.0

        n_lead = 0
        min_fwd = float('inf')
        for box in ann.boxes:
            # box is a numpy array [x, y, z, l, w, h, yaw]
            fwd = float(box[0])   # x = forward
            lat = float(box[1])   # y = lateral (left positive)
            if LEAD_FWD_MIN < fwd < LEAD_FWD_MAX and abs(lat) < LEAD_LAT_MAX:
                n_lead += 1
                if fwd < min_fwd:
                    min_fwd = fwd

        if min_fwd == float('inf'):
            min_fwd = 0.0
        return n_lead > 0, n_lead, min_fwd
    except Exception as e:
        return False, 0, 0.0


def check_bev(agent, scene):
    """Try BEV perception. Returns (seg_ok, n_det)."""
    try:
        agent_input = scene.get_agent_input()
        with torch.no_grad():
            _, data_info = agent._build_solver_inputs(agent_input)
        seg_probs, det_centers = agent._get_bev_perception(data_info, max_voxels=MAX_VOXELS)
        if seg_probs is None:
            return False, 0
        return True, len(det_centers)
    except Exception:
        return False, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-scan', type=int, default=300)
    parser.add_argument('--top',    type=int, default=10)
    parser.add_argument('--seed',   type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('[scene_search_leadveh] Loading navtest tokens ...')
    all_tokens = get_navtest_tokens()
    print(f'  Total: {len(all_tokens)} tokens')
    scan_tokens = random.sample(all_tokens, min(args.n_scan, len(all_tokens)))

    print('[scene_search_leadveh] Loading model (for BEV check) ...')
    agent = build_agent()

    results = []  # (score, n_lead, min_fwd, n_det, token)

    for i, token in enumerate(scan_tokens):
        print(f'[{i+1:03d}/{len(scan_tokens)}] {token}', end=' ', flush=True)
        try:
            scene = load_scene(token)
        except Exception as e:
            print(f'load_err')
            continue

        # GT lead vehicle check
        gt_fwd = 0.0
        try:
            gt_poses = scene.get_future_trajectory().poses[:, :2]
            gt_fwd = float(gt_poses[-1, 0])
        except Exception:
            pass

        if gt_fwd < 1.0:
            print(f'static(gt={gt_fwd:.1f}m)')
            continue

        has_lead, n_lead, min_fwd_lead = has_lead_vehicle(scene)

        if not has_lead:
            print(f'no_lead  gt={gt_fwd:.1f}m')
            continue

        # BEV perception check
        seg_ok, n_det = check_bev(agent, scene)
        status = f'seg={"OK" if seg_ok else "FAIL"}  bev_det={n_det}'
        print(f'LEAD✓ n={n_lead} @{min_fwd_lead:.1f}m  gt={gt_fwd:.1f}m  {status}')

        if seg_ok:
            score = n_lead * 10 + n_det * 5 + gt_fwd * 0.1
            results.append((score, n_lead, min_fwd_lead, n_det, token))

    print(f'\n{"="*65}')
    print(f'Top-{args.top} scenes (lead vehicle + seg/det works):')
    results.sort(key=lambda x: -x[0])
    for score, n_lead, min_fwd, n_det, tok in results[:args.top]:
        print(f'  {tok}  lead={n_lead}@{min_fwd:.1f}m  bev_det={n_det}  score={score:.1f}')
    print(f'{"="*65}')

    if results:
        print(f"\nBest token: '{results[0][4]}'")


if __name__ == '__main__':
    main()
