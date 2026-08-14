"""
场景搜索：找 K=5 轨迹 lateral spread 最大的 token。

扫描 navtest 数据集中的随机 tokens，用温度采样（temperature>1）运行
K=5 DFM，计算轨迹的横向扩散度，输出 top-N 最优场景。

Usage:
  
  python scripts/visualization/scene_search.py [--n-scan 80] [--top 5] [--temp 2.5]
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
SEG_DET_CKPT = str(ROOT / 'output/pretrain/seg_det/checkpoint-20000/model.safetensors')
FUDOKI       = str(ROOT / 'pretrained_model/fudoki')
TEXT_EMB     = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt')
IMAGE_EMB    = str(ROOT / 'pretrained_model/fudoki/image_embedding.pt')
HEADING_MLP  = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt')
BEV_CKPT     = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')

K_RUNS    = 5
DFM_STEPS = 10


# ── Agent 构建 ────────────────────────────────────────────────────────────────

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


# ── 场景加载 ──────────────────────────────────────────────────────────────────

def load_scene(token: str):
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='search'):
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


def get_agent_input(scene, agent=None):
    return scene.get_agent_input()


# ── K 轨迹采样（带温度） ──────────────────────────────────────────────────────

def sample_k_trajs(agent, agent_input, temperature=2.0):
    """Return list of K xy arrays [8,2], or None if error.

    BEV perception is intentionally skipped here — we only need DFM diversity
    for scene selection.  seg/det will be applied in the final ablation_2x2.py.
    """
    # Build solver inputs once (deterministic data info)
    try:
        with torch.no_grad():
            _, data_info_base = agent._build_solver_inputs(agent_input)
    except Exception as e:
        print(f'ERROR (build_solver_inputs): {e}')
        return None

    xys = []
    for _ in range(K_RUNS):
        try:
            with torch.no_grad():
                x_init, data_info = agent._build_solver_inputs(agent_input)
                final_state = agent.solver.sample(
                    x_init=x_init,
                    step_size=1.0 / DFM_STEPS,
                    return_intermediates=False,
                    div_free=0,
                    dtype_categorical=torch.float32,
                    temperature=temperature,
                    datainfo=data_info,
                    cfg_scale=0,
                )
            traj = agent._tokens_to_trajectory(final_state[0])
            xys.append(traj.poses[:, :2].copy())
        except Exception as e:
            print(f'  run error: {e}')

    return xys if len(xys) == K_RUNS else None


# ── 多样性评分 ────────────────────────────────────────────────────────────────

def diversity_score(xys):
    """
    评分 = 跨运行横向标准差（cross-run lateral std）
    - 对每个时间步计算 K 条轨迹的横向位置标准差，取最大值
    - 有效条件：前向位移 [1, 60] m

    这直接衡量"K 条轨迹之间有多不同"，而不是单条轨迹自身的弯曲。
    """
    ends = np.array([xy[-1] for xy in xys])     # [K, 2]
    fwd_mean = ends[:, 0].mean()

    # 合理前向范围（排除停车和倒车/乱码）
    if fwd_mean < 1.0 or fwd_mean > 60.0:
        return 0.0, 0.0, fwd_mean

    # [K, T, 2] → 跨运行横向 std，每个时间步
    traj_lat = np.array([xy[:, 1] for xy in xys])   # [K, 8]
    cross_run_std = traj_lat.std(axis=0)             # [8] std across K runs
    max_cross_std = cross_run_std.max()
    mean_cross_std = cross_run_std.mean()

    # lat_spread = 末端 K 轨迹横向最大差
    lat_spread = traj_lat[:, -1].max() - traj_lat[:, -1].min()

    score = max_cross_std + 0.5 * mean_cross_std
    return score, lat_spread, fwd_mean


# ── 获取 navtest token 列表 ───────────────────────────────────────────────────

def get_navtest_tokens():
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='search_all'):
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


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-scan', type=int, default=80, help='扫描 token 数量')
    parser.add_argument('--top',    type=int, default=8,  help='输出 top-N')
    parser.add_argument('--temp',   type=float, default=2.5, help='采样温度')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'[scene_search] 加载 navtest token 列表 ...')
    all_tokens = get_navtest_tokens()
    print(f'  共 {len(all_tokens)} 个 token')

    scan_tokens = random.sample(all_tokens, min(args.n_scan, len(all_tokens)))

    print(f'[scene_search] 加载模型 ...')
    agent = build_agent()

    results = []  # list of (score, lat_spread, fwd_mean, token)

    for i, token in enumerate(scan_tokens):
        print(f'[{i+1:03d}/{len(scan_tokens)}] token={token}', end=' ', flush=True)
        try:
            scene = load_scene(token)
            agent_input = get_agent_input(scene, agent)
            xys = sample_k_trajs(agent, agent_input, temperature=args.temp)
            if xys is None:
                print('SKIP (error)')
                continue
            score, lat_sp, fwd_m = diversity_score(xys)
            print(f'score={score:.3f}  lat_spread={lat_sp:.2f}m  fwd={fwd_m:.1f}m')
            results.append((score, lat_sp, fwd_m, token))
        except Exception as e:
            print(f'ERROR: {e}')

    # 排序输出
    results.sort(key=lambda x: -x[0])
    print(f'\n{"="*60}')
    print(f'Top-{args.top} 多样性场景（temperature={args.temp}）：')
    print(f'{"rank":<5} {"token":<20} {"score":>7} {"lat_spread":>11} {"fwd":>7}')
    print(f'{"-"*60}')
    for rank, (score, lat_sp, fwd_m, token) in enumerate(results[:args.top], 1):
        print(f'{rank:<5} {token:<20} {score:>7.3f} {lat_sp:>10.2f}m {fwd_m:>6.1f}m')
    print(f'{"="*60}')
    print(f'\n用法示例（token 填入 ablation_2x2.py 中 DEFAULT_TOKEN）：')
    if results:
        print(f"  DEFAULT_TOKEN = '{results[0][3]}'")


if __name__ == '__main__':
    main()
