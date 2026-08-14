"""
Visualise all DFM intermediate steps for both models on one scene.

Layout: 2 rows (Our model / Traditional DFM)
        K+1 columns (step 0 = noise init, step 1..K)

Usage:
  python scripts/visualization/visualize_dfm_steps.py [--token TOKEN] [--steps N]
"""

import os, sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['LOCAL_RANK']            = '0'
os.environ['OPENSCENE_DATA_ROOT']   = '/path/to/dataset'
os.environ['NUPLAN_MAPS_ROOT']      = '/path/to/dataset/maps'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/path/to/dataset')
OUT_DIR   = ROOT / 'exp/ablation_vis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUR_CKPT     = str(ROOT / 'output/train/navsim_bev/checkpoint-10000')
TRAD_CKPT    = str(ROOT / 'pretrained_model/cmmi')
BEV_CKPT     = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth')
SEG_DET_CKPT = str(ROOT / 'output/pretrain/seg_det/checkpoint-20000/model.safetensors')
FUDOKI       = str(ROOT / 'pretrained_model/fudoki')
TEXT_EMB     = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt')
IMAGE_EMB    = str(ROOT / 'pretrained_model/fudoki/image_embedding.pt')
HEADING_MLP  = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt')

DEFAULT_TOKEN = '0179d579d30e588c'
DEFAULT_STEPS = 10
COLOR_GT   = '#59a14f'
COLOR_TRAJ = '#e31a1c'


def load_scene(token):
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    GlobalHydra.instance().clear()
    cfg = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg, version_base=None, job_name='steps_vis'):
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


def get_all_intermediates(agent, agent_input, steps):
    """Return list of [8,2] arrays for steps 0..K (0 = noise init)."""
    with torch.no_grad():
        x_init, data_info = agent._build_solver_inputs(agent_input)
        intermediates = agent.solver.sample(
            x_init=x_init,
            step_size=1.0 / steps,
            return_intermediates=True,
            div_free=0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=0,
        )
    xys = []
    for k in range(steps + 1):
        try:
            t = agent._tokens_to_trajectory(intermediates[k])
            xys.append(t.poses[:, :2].copy())
        except Exception:
            xys.append(None)
    return xys


def draw_cell(ax, scene, cur_frame, gt_poses, xy, label, color=COLOR_TRAJ):
    from navsim.visualization.bev import add_configured_bev_on_ax
    add_configured_bev_on_ax(ax, scene.map_api, cur_frame)
    # GT trajectory
    gt_full = np.vstack([[0, 0], gt_poses])
    ax.plot(gt_full[:, 1], gt_full[:, 0],
            color=COLOR_GT, lw=1.8, ls='--', alpha=0.8, zorder=8)
    # Ego marker
    ax.plot(0, 0, marker='^', color='black', ms=7, zorder=15)

    if xy is not None:
        # Always prepend ego origin so every trajectory starts from (0,0)
        pts = np.vstack([[0., 0.], xy])          # [9, 2]
        fwd, lat = xy[-1, 0], xy[-1, 1]
        # Draw regardless of range — clip the axes to BEV window instead
        ax.plot(pts[:, 1], pts[:, 0], color=color, lw=2.2, alpha=0.9, zorder=10,
                solid_capstyle='round', solid_joinstyle='round')
        # Endpoint dot only if within view
        if abs(fwd) < 100 and abs(lat) < 50:
            ax.scatter(lat, fwd, color=color, s=28, zorder=11)
        info = f'({fwd:.1f},{lat:.2f})'
        if abs(fwd) > 100 or abs(lat) > 50:
            info += '\n(OOB)'
    else:
        info = 'decode err'
    ax.set_xlim(14, -14)
    ax.set_ylim(-2, 26)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_title(f'{label}\n{info}', fontsize=7, pad=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--token', default=DEFAULT_TOKEN)
    p.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    p.add_argument('--out', default=str(OUT_DIR / 'dfm_steps.png'))
    args = p.parse_args()
    K = args.steps

    print(f'Token: {args.token}   DFM steps: {K}')
    scene = load_scene(args.token)
    agent_input = scene.get_agent_input()
    cur_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    gt_poses  = scene.get_future_trajectory().poses[:, :2]

    # ── Our model ─────────────────────────────────────────────────────────────
    print('Loading OUR model ...')
    from navsim.agents.cmmi_bev_agent import CMMIBevAgent
    our = CMMIBevAgent(
        fudoki_path=FUDOKI, cmmi_path=OUR_CKPT, bev_ckpt_path=BEV_CKPT,
        text_embedding_path=TEXT_EMB, image_embedding_path=IMAGE_EMB,
        heading_mlp_path=HEADING_MLP, discrete_fm_steps=K,
    )
    our.initialize()
    print(f'Running {K} DFM steps (return_intermediates=True) ...')
    xys_our = get_all_intermediates(our, agent_input, K)
    del our; torch.cuda.empty_cache()

    # ── Traditional ───────────────────────────────────────────────────────────
    print('Loading TRADITIONAL model ...')
    from navsim.agents.cmmi_hybrid_agent import CMMIHybridAgent
    trad = CMMIHybridAgent(
        fudoki_path=FUDOKI, cmmi_path=TRAD_CKPT, bev_ckpt_path=BEV_CKPT,
        seg_det_ckpt_path=SEG_DET_CKPT, text_embedding_path=TEXT_EMB,
        image_embedding_path=IMAGE_EMB, heading_mlp_path=HEADING_MLP,
        discrete_fm_steps=K,
    )
    trad.initialize()
    print(f'Running {K} DFM steps (return_intermediates=True) ...')
    xys_trad = get_all_intermediates(trad, agent_input, K)
    del trad; torch.cuda.empty_cache()

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\nStep-by-step decoded trajectories (fwd=x[-1,0], lat=x[-1,1]):')
    print(f'{"Step":>5}  {"OUR fwd":>9}  {"OUR lat":>9}  {"TRAD fwd":>9}  {"TRAD lat":>9}')
    for k in range(K + 1):
        o = xys_our[k];  t = xys_trad[k]
        o_str = f'{o[-1,0]:9.3f}  {o[-1,1]:9.3f}' if o is not None else '      ERR        ERR'
        t_str = f'{t[-1,0]:9.3f}  {t[-1,1]:9.3f}' if t is not None else '      ERR        ERR'
        print(f'{k:5d}  {o_str}  {t_str}')

    # ── Figure: 2 rows × (K+1) cols ──────────────────────────────────────────
    ncols = K + 1
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.5, 2 * 5),
                             facecolor='white')

    row_labels = [f'Our model\n(§3.2 checkpoint-10000)', f'Traditional DFM\n(pretrained)']
    colors     = ['#e31a1c', '#1f78b4']

    for row, (xys, rlabel, col) in enumerate(zip([xys_our, xys_trad], row_labels, colors)):
        for k in range(K + 1):
            ax = axes[row, k]
            step_label = f'step {k}' if k > 0 else 'init (noise)'
            draw_cell(ax, scene, cur_frame, gt_poses, xys[k], step_label, color=col)

        # Row label on leftmost cell
        axes[row, 0].set_ylabel(rlabel, fontsize=9, labelpad=4)

    fig.suptitle(
        f'DFM Intermediate Steps | token={args.token} | K={K}\n'
        f'Green dashed = GT trajectory   |   Coloured = decoded trajectory at each denoising step',
        fontsize=10, y=1.01,
    )

    plt.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
