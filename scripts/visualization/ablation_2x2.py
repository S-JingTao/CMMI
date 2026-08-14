"""
2x2 Ablation Visualization for Section 3.2 x Section 3.3.

Layout (matches visualize_3method_comparison.py style):
  Pure 2x2 BEV grid: CubicSpline trajectories, set_aspect='auto', 1.2px spines.

  TL: Our multi-traj Sec3.2   (K diverse raw trajectories)
  TR: Ours Sec3.2 + Sec3.3    (K LM-refined, before argmax)
  BL: Traditional DFM          (K runs, CE loss, mode collapse)
  BR: Traditional DFM + Sec3.3 (K LM-refined)

Usage (from cmmi root, cmmi env):
  python scripts/visualization/ablation_2x2.py [--token TOKEN] [--out PATH]
"""

import os, sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['LOCAL_RANK']            = '0'
os.environ['OPENSCENE_DATA_ROOT']   = '/path/to/dataset'
os.environ['NUPLAN_MAPS_ROOT']      = '/path/to/dataset/maps'

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/path/to/dataset')
OUT_DIR   = ROOT / 'exp/ablation_vis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
OUR_CKPT     = str(ROOT / 'output/train/navsim_bev_lplan5/checkpoint-10000')
TRAD_CKPT    = str(ROOT / 'pretrained_model/cmmi')
BEV_CKPT     = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')
SEG_DET_CKPT = str(ROOT / 'output/pretrain/seg_det/checkpoint-20000/model.safetensors')
FUDOKI       = str(ROOT / 'pretrained_model/fudoki')
TEXT_EMB     = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt')
IMAGE_EMB    = str(ROOT / 'pretrained_model/fudoki/image_embedding.pt')
HEADING_MLP  = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt')

DEFAULT_TOKEN = '483042d5dc175e99'

# ── Inference hyper-parameters ────────────────────────────────────────────────
K_RUNS              = 5
N_POOL              = 25
DFM_STEPS_OURS      = 10   # our model (L_plan training)
DFM_STEPS_TRAD      = 5    # original baseline default
TEMPERATURE         = 1.2
MAX_VOXELS_VIS      = 20000

# ── Visual style (matches visualize_3method_comparison.py) ───────────────────
CELL        = 5.5
SPINE_LW    = 3.0
GT_COLOR    = '#59a14f'
TRAJ_COLORS = ['#1f78b4', '#33a02c', '#ff7f00', '#e31a1c', '#6a3d9a']

BEV_XLIM = (30, -30)
BEV_YLIM = (-5, 55)

COL_LABELS = ['Raw (K diverse traj.)', 'LM-refined (+Sec. 3.3)']
ROW_LABELS  = ['Ours ($L_{plan}$)', 'Traditional DFM']


# ── GT obstacle extraction ───────────────────────────────────────────────────

def extract_gt_det_centers(scene, fwd_min=1.0, fwd_max=40.0, lat_max=8.0):
    try:
        hist = scene.scene_metadata.num_history_frames
        ann  = scene.frames[hist - 1].annotations
        if ann is None or not hasattr(ann, 'boxes') or len(ann.boxes) == 0:
            return []
        centers = []
        for box in ann.boxes:
            fwd, lat = float(box[0]), float(box[1])
            if fwd_min < fwd < fwd_max and abs(lat) < lat_max:
                centers.append((fwd, lat))
        return centers
    except Exception:
        return []


# ── Scene loading ─────────────────────────────────────────────────────────────

def load_scene(token: str):
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='ablation'):
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


# ── Inference ─────────────────────────────────────────────────────────────────

def traj_dist(xy_a, xy_b):
    return float(np.linalg.norm(xy_a - xy_b, axis=1).mean())


def greedy_diverse_select(pool, k):
    if len(pool) <= k:
        return list(range(len(pool)))
    pool_arr      = np.stack(pool)
    mean_traj     = pool_arr.mean(axis=0)
    dists_to_mean = [traj_dist(xy, mean_traj) for xy in pool]
    selected      = [int(np.argmax(dists_to_mean))]
    while len(selected) < k:
        min_dists = []
        for i, xy in enumerate(pool):
            if i in selected:
                min_dists.append(-1.0)
                continue
            d = min(traj_dist(xy, pool[j]) for j in selected)
            min_dists.append(d)
        selected.append(int(np.argmax(min_dists)))
    return selected


def get_k_independent_trajs(agent, agent_input, gt_det_centers=None, dfm_steps=10):
    with torch.no_grad():
        _, data_info_init = agent._build_solver_inputs(agent_input)
    seg_probs, bev_det_centers = agent._get_bev_perception(
        data_info_init, max_voxels=MAX_VOXELS_VIS
    )
    if gt_det_centers is not None and len(gt_det_centers) > 0:
        det_centers = gt_det_centers
        print(f'  [Sec3.3] seg={"OK" if seg_probs is not None else "FAIL"}, '
              f'det={len(det_centers)} GT vehicles')
    else:
        det_centers = bev_det_centers
        print(f'  [Sec3.3] seg={"OK" if seg_probs is not None else "unavailable"}, '
              f'det={len(det_centers)} (BEV)')

    pool = []
    print(f'  Sampling {N_POOL} DFM trajectories ({dfm_steps} steps each) ...')
    for run_idx in range(N_POOL):
        try:
            with torch.no_grad():
                x_init, data_info = agent._build_solver_inputs(agent_input)
                final_state = agent.solver.sample(
                    x_init=x_init,
                    step_size=1.0 / dfm_steps,
                    return_intermediates=False,
                    div_free=0,
                    dtype_categorical=torch.float32,
                    temperature=TEMPERATURE,
                    datainfo=data_info,
                    cfg_scale=0,
                )
            traj = agent._tokens_to_trajectory(final_state[0])
            pool.append(traj.poses[:, :2].copy())
        except Exception as e:
            print(f'  run {run_idx+1} error: {e}')

    def is_valid(xy):
        if np.any(np.abs(xy[:, 1]) > 15.0): return False
        if xy[-1, 0] < 0.5: return False
        if np.linalg.norm(np.diff(xy, axis=0), axis=1).max() > 8.0: return False
        return True

    valid_pool = [xy for xy in pool if is_valid(xy)] or pool
    lats = [xy[-1, 1] for xy in valid_pool]
    print(f'  Valid: {len(valid_pool)}  spread={max(lats)-min(lats):.2f}m')

    idxs     = greedy_diverse_select(valid_pool, K_RUNS)
    raw_xys  = [valid_pool[i] for i in idxs]
    sel_lats = [raw_xys[i][-1, 1] for i in range(len(raw_xys))]
    print(f'  Selected lats={[f"{v:+.2f}" for v in sel_lats]}'
          f'  spread={max(sel_lats)-min(sel_lats):.2f}m')

    refined_xys = [
        _smooth1d(agent._refine_trajectory(xy, seg_probs, det_centers))
        for xy in raw_xys
    ]
    return raw_xys, refined_xys


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _smooth1d(xy, sigma=1.0):
    return np.stack([
        gaussian_filter1d(xy[:, 0], sigma=sigma),
        gaussian_filter1d(xy[:, 1], sigma=sigma),
    ], axis=1)


def _spline(poses):
    all_pts = np.vstack([[0., 0.], poses[:, :2]])
    n = len(all_pts)
    t = np.arange(n, dtype=float)
    t_f = np.linspace(0, n - 1, 200)
    xs = CubicSpline(t, all_pts[:, 0])(t_f)
    ys = CubicSpline(t, all_pts[:, 1])(t_f)
    return np.stack([xs, ys], axis=1)


def draw_bev_traj(ax, poses_xy, color, lw=2.5):
    if poses_xy is None or len(poses_xy) < 2:
        return
    spts = _spline(poses_xy)
    ax.plot(spts[:, 1], spts[:, 0],
            color=color, linewidth=lw, zorder=5,
            solid_capstyle='round', solid_joinstyle='round')
    ax.plot(poses_xy[:, 1], poses_xy[:, 0],
            marker='o', markersize=3.5, linestyle='none',
            color=color, markeredgecolor='black', markeredgewidth=0.4, zorder=6)


def draw_obstacle_boxes(ax, gt_det_centers, scene):
    if not gt_det_centers:
        return
    try:
        hist = scene.scene_metadata.num_history_frames
        ann  = scene.frames[hist - 1].annotations
        if ann is None or not hasattr(ann, 'boxes'):
            return
        for box in ann.boxes:
            fwd, lat = float(box[0]), float(box[1])
            length, width, yaw = float(box[3]), float(box[4]), float(box[6])
            if not any(abs(fwd - cf) < 1.0 and abs(lat - cl) < 1.0
                       for cf, cl in gt_det_centers):
                continue
            corners = np.array([[ length/2,  width/2],
                                 [ length/2, -width/2],
                                 [-length/2, -width/2],
                                 [-length/2,  width/2]])
            c, s = np.cos(yaw), np.sin(yaw)
            corners = corners @ np.array([[c, -s], [s, c]]).T + np.array([fwd, lat])
            ax.add_patch(mpatches.Polygon(
                corners[:, [1, 0]],
                closed=True, edgecolor='red', facecolor='red',
                alpha=0.30, linewidth=2.0, zorder=12,
            ))
    except Exception as e:
        print(f'  [draw_obstacle_boxes] {e}')


# ── Main render ───────────────────────────────────────────────────────────────

def render(scene, tl_raw, tl_ref, bl_raw, bl_ref,
           out_path: Path, gt_det_centers=None):
    from navsim.visualization.bev import add_configured_bev_on_ax

    hist      = scene.scene_metadata.num_history_frames
    cur_frame = scene.frames[hist - 1]
    gt_poses  = scene.get_future_trajectory().poses[:, :2]

    for name, trajs in [('TL-raw', tl_raw), ('TL-ref', tl_ref),
                        ('BL-raw', bl_raw), ('BL-ref', bl_ref)]:
        lats = [xy[-1, 1] for xy in trajs]
        print(f'  {name} spread={max(lats)-min(lats):.2f}m  '
              f'lats={[f"{v:+.2f}" for v in lats]}')

    # Figure: 2x2, high-res
    fig = plt.figure(figsize=(CELL * 2 * 2, CELL * 2 * 2), facecolor='white')

    # Equal gaps on all four sides; hspace == wspace for visually uniform spacing
    outer = gridspec.GridSpec(
        2, 2,
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.995, top=0.955, bottom=0.02,
    )

    cells = [
        (0, 0, tl_raw, False),
        (0, 1, tl_ref, True),
        (1, 0, bl_raw, False),
        (1, 1, bl_ref, True),
    ]

    for gs_row, gs_col, trajs, is_refined in cells:
        ax = fig.add_subplot(outer[gs_row, gs_col])
        add_configured_bev_on_ax(ax, scene.map_api, cur_frame)

        # GT trajectory (green dashed)
        spts_gt = _spline(gt_poses)
        ax.plot(spts_gt[:, 1], spts_gt[:, 0],
                color=GT_COLOR, linewidth=2.0, linestyle='--',
                alpha=0.9, zorder=8)

        # K candidate trajectories
        for i, xy in enumerate(trajs):
            draw_bev_traj(ax, xy, color=TRAJ_COLORS[i % len(TRAJ_COLORS)])

        # Obstacle boxes: draw in all 4 panels for visual consistency
        draw_obstacle_boxes(ax, gt_det_centers, scene)

        # Ego marker
        ax.plot(0, 0, marker='^', color='black', markersize=10, zorder=15,
                markeredgecolor='white', markeredgewidth=0.8)

        # BEV range + no white-padding
        ax.set_xlim(*BEV_XLIM)
        ax.set_ylim(*BEV_YLIM)
        ax.set_aspect('auto')

        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)

        # Column header and row labels removed (text moved inside or to caption)

        # In-panel legend: only in TL (top-left) cell
        if gs_row == 0 and gs_col == 0:
            leg_handles = [
                Line2D([0], [0], color=TRAJ_COLORS[i], linewidth=2.5,
                       marker='o', markersize=5, markeredgecolor='black',
                       markeredgewidth=0.4, label=f'Step-{i+1} Traj.')
                for i in range(K_RUNS)
            ] + [
                Line2D([0], [0], color=GT_COLOR, linewidth=2.0,
                       linestyle='--', label='G.T. Traj.')
            ]
            # Width ~1/4 of panel: anchor to upper-right 25% of axes width
            ax.legend(
                handles=leg_handles,
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                fontsize=28, frameon=False,
                borderpad=0.8,
                handlelength=2.0, handletextpad=0.6,
                labelspacing=0.4,
            )

    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved -> {out_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--token', default=None)
    p.add_argument('--out',   default=str(OUT_DIR / 'ablation_2x2.png'))
    args = p.parse_args()

    token = args.token or DEFAULT_TOKEN
    print(f'Scene token: {token}')

    print('Loading scene ...')
    scene       = load_scene(token)
    agent_input = scene.get_agent_input()

    gt_det_centers = extract_gt_det_centers(scene)
    if gt_det_centers:
        print(f'GT det centers: {len(gt_det_centers)} vehicles '
              f'{[(f"{f:.1f}m", f"{l:+.1f}m") for f,l in gt_det_centers]}')
    else:
        print('GT det centers: none in range')

    # Our model
    print('\n[OUR MODEL] CMMIBevAgent ck-10000 (Sec3.2 L_plan)')
    from navsim.agents.cmmi_bev_agent import CMMIBevAgent
    our_agent = CMMIBevAgent(
        fudoki_path=FUDOKI, cmmi_path=OUR_CKPT, bev_ckpt_path=BEV_CKPT,
        text_embedding_path=TEXT_EMB, image_embedding_path=IMAGE_EMB,
        heading_mlp_path=HEADING_MLP, discrete_fm_steps=DFM_STEPS_OURS,
    )
    our_agent.initialize()
    tl_raw, tl_ref = get_k_independent_trajs(our_agent, agent_input,
                                              gt_det_centers=gt_det_centers,
                                              dfm_steps=DFM_STEPS_OURS)
    del our_agent; torch.cuda.empty_cache()

    # Traditional DFM
    print('\n[TRADITIONAL DFM] CMMIHybridAgent pretrained (CE loss)')
    from navsim.agents.cmmi_hybrid_agent import CMMIHybridAgent
    trad_agent = CMMIHybridAgent(
        fudoki_path=FUDOKI, cmmi_path=TRAD_CKPT, bev_ckpt_path=BEV_CKPT,
        seg_det_ckpt_path=SEG_DET_CKPT,
        text_embedding_path=TEXT_EMB, image_embedding_path=IMAGE_EMB,
        heading_mlp_path=HEADING_MLP, discrete_fm_steps=DFM_STEPS_TRAD,
    )
    trad_agent.initialize()
    bl_raw, bl_ref = get_k_independent_trajs(trad_agent, agent_input,
                                              gt_det_centers=gt_det_centers,
                                              dfm_steps=DFM_STEPS_TRAD)
    del trad_agent; torch.cuda.empty_cache()

    print('\nRendering ...')
    render(scene, tl_raw, tl_ref, bl_raw, bl_ref,
           Path(args.out), gt_det_centers=gt_det_centers)


if __name__ == '__main__':
    main()
