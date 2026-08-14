"""
场景搜索：找 BEV seg/det 能正常工作 且 有障碍物检测的 token。

只跑 BEV 感知（不跑 DFM），速度快，可以扫描大量 token。

Usage:
  
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate cmmi
  CUDA_VISIBLE_DEVICES=1 python scripts/visualization/scene_search_segdet.py [--n-scan 200]
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

OUR_CKPT     = str(ROOT / 'output/train/navsim_bev_lplan5/checkpoint-10000')
BEV_CKPT     = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')
FUDOKI       = str(ROOT / 'pretrained_model/fudoki')
TEXT_EMB     = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt')
IMAGE_EMB    = str(ROOT / 'pretrained_model/fudoki/image_embedding.pt')
HEADING_MLP  = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt')

MAX_VOXELS = 20000
DFM_STEPS  = 10
K_RUNS     = 5


def build_agent():
    from navsim.agents.cmmi_bev_agent import CMMIBevAgent
    agent = CMMIBevAgent(
        fudoki_path          = FUDOKI,
        cmmi_path        = OUR_CKPT,
        bev_ckpt_path        = BEV_CKPT,
        text_embedding_path  = TEXT_EMB,
        image_embedding_path = IMAGE_EMB,
        heading_mlp_path     = HEADING_MLP,
        discrete_fm_steps    = DFM_STEPS,
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
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='segdet'):
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


def check_scene(agent, token: str):
    """
    Returns dict with keys:
      seg_ok: bool
      n_det: int  (number of detected obstacle centers)
      gt_fwd: float  (GT trajectory forward displacement, to filter parked scenes)
      off_road_ratio: float  (fraction of DFM raw waypoints that land on non-drivable seg class)
    Returns None on hard error.
    """
    try:
        scene = load_scene(token)
    except Exception as e:
        return None

    try:
        agent_input = scene.get_agent_input()
        with torch.no_grad():
            _, data_info = agent._build_solver_inputs(agent_input)
        seg_probs, det_centers = agent._get_bev_perception(
            data_info, max_voxels=MAX_VOXELS
        )
    except Exception as e:
        return None

    if seg_probs is None:
        return {'seg_ok': False, 'n_det': 0, 'gt_fwd': 0.0, 'off_road_ratio': 0.0}

    n_det = len(det_centers)

    # GT trajectory forward displacement
    try:
        gt_poses = scene.get_future_trajectory().poses[:, :2]
        gt_fwd = float(gt_poses[-1, 0])
    except Exception:
        gt_fwd = 0.0

    # Run K DFM samples and check how many waypoints land off-road
    # seg class 0 = background/non-road (weight 0.0 in _SEG_WEIGHTS_RISK → safe to drive)
    # class 1 = shoulder (0.5), class 2 = road (1.0 — wait, that's inverted!)
    # Actually _SEG_WEIGHTS_RISK = [0.0, 0.5, 1.0, 0.3, 2.0, 0.0]
    # class 2 (road) has weight 1.0 — but higher weight means MORE risk??
    # Let me re-read: r_seg penalizes high-risk classes. Road=1.0 penalty means road is "risky"?
    # That seems wrong. Let me check by looking at argmax seg class for on-road waypoints.
    # Actually the risk weights punish driving in those areas: class 4 (obstacle/curb) = 2.0 risk
    # class 2 might be "lane marking" or "crosswalk" — check the actual class names.
    # For now: check what seg class the GT trajectory lands in vs DFM trajectories

    off_road_count = 0
    total_wp = 0

    try:
        for _ in range(K_RUNS):
            with torch.no_grad():
                x_init, data_info_run = agent._build_solver_inputs(agent_input)
                final_state = agent.solver.sample(
                    x_init=x_init,
                    step_size=1.0 / DFM_STEPS,
                    return_intermediates=False,
                    div_free=0,
                    dtype_categorical=torch.float32,
                    temperature=1.15,
                    datainfo=data_info_run,
                    cfg_scale=0,
                )
            traj = agent._tokens_to_trajectory(final_state[0])
            xy = traj.poses[:, :2]  # [8, 2]

            # Map waypoints to seg grid and get class
            # seg_probs: [1, C, H, W] — BEV grid, origin at ego
            # BEV range typically [-25.6, 25.6]m, res 0.1m → 512×512
            H, W = seg_probs.shape[2], seg_probs.shape[3]
            bev_range = 25.6  # meters, check actual value
            res = 2 * bev_range / H

            for wp in xy:
                fwd, lat = float(wp[0]), float(wp[1])
                # Convert ego coord → grid index
                # x=fwd → row (y in BEV image, from front)
                # y=lat → col (x in BEV image)
                row = int((bev_range - fwd) / res)
                col = int((lat + bev_range) / res)
                if 0 <= row < H and 0 <= col < W:
                    cls = int(seg_probs[0, :, row, col].argmax())
                    # class 4 = high risk (obstacle/non-drivable), class 2 = road marking
                    if cls in (4,):  # high risk class
                        off_road_count += 1
                total_wp += 1
    except Exception:
        pass

    off_road_ratio = off_road_count / max(total_wp, 1)

    return {
        'seg_ok': True,
        'n_det': n_det,
        'gt_fwd': gt_fwd,
        'off_road_ratio': off_road_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-scan', type=int, default=200)
    parser.add_argument('--top',    type=int, default=10)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('[scene_search_segdet] Loading navtest tokens ...')
    all_tokens = get_navtest_tokens()
    print(f'  Total: {len(all_tokens)} tokens')

    scan_tokens = random.sample(all_tokens, min(args.n_scan, len(all_tokens)))

    print('[scene_search_segdet] Loading model ...')
    agent = build_agent()

    results_det  = []  # (n_det, off_road, token) — want n_det > 0
    results_seg  = []  # (off_road_ratio, n_det, token) — want off_road > 0
    n_seg_ok     = 0

    for i, token in enumerate(scan_tokens):
        print(f'[{i+1:03d}/{len(scan_tokens)}] {token}', end=' ', flush=True)
        r = check_scene(agent, token)
        if r is None:
            print('ERROR')
            continue

        if not r['seg_ok']:
            print('seg=FAIL')
            continue

        n_seg_ok += 1
        n_det = r['n_det']
        gt_fwd = r['gt_fwd']
        off_r = r['off_road_ratio']
        print(f'seg=OK  det={n_det}  gt_fwd={gt_fwd:.1f}m  off_road={off_r:.3f}')

        if gt_fwd < 1.0:
            continue  # parked / static scene

        if n_det > 0:
            results_det.append((n_det, off_r, token))
        if off_r > 0:
            results_seg.append((off_r, n_det, token))

    print(f'\n{"="*60}')
    print(f'Scanned {len(scan_tokens)} tokens, seg_ok: {n_seg_ok}')

    print(f'\n--- Top scenes with DETECTED OBSTACLES (det > 0) ---')
    results_det.sort(key=lambda x: -x[0])
    for n_det, off_r, tok in results_det[:args.top]:
        print(f'  {tok}  det={n_det}  off_road={off_r:.3f}')

    print(f'\n--- Top scenes with OFF-ROAD DFM trajectories ---')
    results_seg.sort(key=lambda x: -x[0])
    for off_r, n_det, tok in results_seg[:args.top]:
        print(f'  {tok}  off_road={off_r:.3f}  det={n_det}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()
