"""
Animated GIF: 4 scenes × 14 frames @ 2Hz.  Layout 2×2:
  row 0: Go Straight  |  Intersection
  row 1: Turn Left    |  Turn Right

Phase 1: 8-GPU parallel inference  (4 scenes × 11 frames = 44 tasks spread across 8 GPUs)
Phase 2: single-process rendering  (matplotlib → Pillow GIF)
Output: exp/demo_gif/demo_4scenes_v3.gif
"""

import os, sys, pickle
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['OPENSCENE_DATA_ROOT'] = '/path/to/dataset'
os.environ['NUPLAN_MAPS_ROOT']    = '/path/to/dataset/maps'

import cv2, numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
import torch
import torch.multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.common.dataclasses import SensorConfig, AgentInput
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.common.enums import BoundingBoxIndex

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/path/to/dataset')
OUT_DIR   = ROOT / 'exp/demo_gif'
TMP_DIR   = ROOT / 'exp/demo_gif/tmp_preds'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

SCENES = [
    ('72d842bc596b536b', 'Go Straight'),
    ('90ed1fb3861c56d7', 'Intersection'),
    ('8429a35187bb5c08', 'Turn Left'),
    ('fd513762a5ea5dd4', 'Turn Right'),
]
N_FRAMES  = 14
CUR_IDX   = 3
CAM_RATIO = 1920 / 1080
COLOR_PRED = '#CC0000'
COLOR_GT   = '#59a14f'
SENSOR_CFG_AGENT = SensorConfig(
    cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[],
    cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
    lidar_pc=[3],
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: parallel inference worker
# ─────────────────────────────────────────────────────────────────────────────
def _make_loader(tokens, sensor_config, job):
    GlobalHydra.instance().clear()
    cfg = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg, version_base=None, job_name=job):
        sf = instantiate(hydra.compose(config_name='navtest'))
    sf.tokens = tokens
    return SceneLoader(
        sensor_blobs_path=DATA_ROOT / 'sensor_blobs/test',
        data_path=DATA_ROOT / 'navsim_logs/test',
        scene_filter=sf,
        sensor_config=sensor_config,
    )

def inference_worker(rank, all_tasks):
    """Runs on GPU `rank`. Saves pred arrays to TMP_DIR/{token}_{t}.pkl."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['LOCAL_RANK'] = '0'
    # suppress hydra re-init noise
    import warnings; warnings.filterwarnings('ignore')

    my_tasks = all_tasks[rank::8]   # interleaved split
    if not my_tasks:
        return

    tokens = list({tok for tok, _ in my_tasks})
    loader = _make_loader(tokens, SensorConfig.build_no_sensors(), f'w{rank}')

    from navsim.agents.cmmi_hybrid_agent import CMMIHybridAgent
    agent = CMMIHybridAgent(
        fudoki_path         = str(ROOT / 'pretrained_model/fudoki'),
        cmmi_path       = str(ROOT / 'pretrained_model/cmmi'),
        bev_ckpt_path       = str(ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth'),
        seg_det_ckpt_path   = str(ROOT / 'output/pretrain/seg_det/checkpoint-20000/model.safetensors'),
        text_embedding_path = str(ROOT / 'pretrained_model/fudoki/text_embedding.pt'),
        image_embedding_path= str(ROOT / 'pretrained_model/fudoki/image_embedding.pt'),
        heading_mlp_path    = str(ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt'),
        discrete_fm_steps   = 5,
    )
    agent.initialize()

    for token, t in my_tasks:
        out = TMP_DIR / f'{token}_{t}.pkl'
        if out.exists():   # skip if already cached
            continue
        frame_dicts = loader.scene_frames_dicts[token]
        start = max(0, t - 3)
        win   = list(frame_dicts[start:t+1])
        while len(win) < 4:
            win = [win[0]] + win
        ai   = AgentInput.from_scene_dict_list(
            win, DATA_ROOT / 'sensor_blobs/test', 4, SENSOR_CFG_AGENT)
        pred = agent.compute_trajectory(ai).poses   # (8,3)
        with open(out, 'wb') as f:
            pickle.dump(pred, f)

    print(f'[GPU {rank}] done {len(my_tasks)} tasks', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def transform_to_frame_t(poses_3, ego_poses_all, t):
    xt, yt, ht = ego_poses_all[t].astype(float)
    ct, st = np.cos(ht), np.sin(ht)
    Rinv = np.array([[ct, st], [-st, ct]])
    r = poses_3.copy()
    r[:, :2] = (Rinv @ (poses_3[:, :2] - np.array([xt, yt])).T).T
    r[:, 2]  = poses_3[:, 2] - ht
    return r

def make_polys(annotations):
    polys = []
    for box in annotations.boxes:
        bx, by = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
        h, l, w = box[BoundingBoxIndex.HEADING], box[BoundingBoxIndex.LENGTH], box[BoundingBoxIndex.WIDTH]
        c, s = np.cos(h), np.sin(h)
        corners = np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])
        polys.append(Polygon((np.array([[c,-s],[s,c]]) @ corners.T).T + [bx, by]))
    return polys

def clip_traj(poses, polys):
    pts = [np.zeros(3, dtype=np.float32)]
    for p in poses:
        if any(poly.contains(Point(p[0], p[1])) for poly in polys):
            break
        pts.append(p)
    return np.array(pts)

def draw_bev_traj(ax, poses, color, lw=2.5):
    if len(poses) < 2: return
    ax.plot(poses[:,1], poses[:,0], color=color, linewidth=lw,
            marker='o', markersize=3.5,
            markeredgecolor='black', markeredgewidth=0.4, zorder=5)

def hex2bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def prepare_cam_image(image):
    img = np.array(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.mean() < 80:
        factor = min(80 / (img.mean() + 1e-3), 3.0)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img

def project_traj(poses, camera):
    K  = np.array(camera.intrinsics)
    Rl = np.array(camera.sensor2lidar_rotation).T
    tl = -Rl @ np.array(camera.sensor2lidar_translation)
    all_xy = np.vstack([[0.,0.], poses[:,:2]])
    seg = np.linalg.norm(np.diff(all_xy, axis=0), axis=1)
    arc = np.concatenate([[0.], np.cumsum(seg)])
    total = arc[-1]
    n = max(len(all_xy)*2, int(total/0.25)+1)
    xs = np.interp(np.linspace(0, total, n), arc, all_xy[:,0])
    ys = np.interp(np.linspace(0, total, n), arc, all_xy[:,1])
    pts = []
    for x, y in zip(xs, ys):
        P = Rl @ np.array([x, y, 0.]) + tl
        if P[2] < 0.1: continue
        pts.append((int(round(K[0,0]*P[0]/P[2] + K[0,2])),
                    int(round(K[1,1]*P[1]/P[2] + K[1,2]))))
    return pts

def draw_cam_traj(img, pts, color_bgr):
    if len(pts) < 2: return
    bH, bW = img.shape[:2]
    for u, v in pts:
        if 0 <= u < bW and 0 <= v < bH:
            cv2.circle(img, (int(u), int(v)), 11, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)),  8, color_bgr,     -1, cv2.LINE_AA)

def render_frame(t, scene_data):
    fig = plt.figure(figsize=(int(2*(1+CAM_RATIO)*5.5), 2*5.5), facecolor='white')
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              wspace=0.04, hspace=0.20,
                              left=0.005, right=0.995,
                              top=0.970, bottom=0.06)

    for idx, (token, label) in enumerate(SCENES):
        row, col = divmod(idx, 2)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row, col],
            width_ratios=[1, CAM_RATIO], wspace=0.02)
        ax_bev = fig.add_subplot(inner[0, 0])
        ax_cam = fig.add_subplot(inner[0, 1])

        d = scene_data[token]
        ego_poses_all  = d['ego_poses_all']
        gt_3           = d['gt_3']
        pred_per_frame = d['pred_per_frame']

        add_configured_bev_on_ax(ax_bev, d['sc_bev'].map_api, d['sc_bev'].frames[t])
        ann   = d['sc_bev'].frames[t].annotations
        polys = make_polys(ann)

        if t >= CUR_IDX:
            pred_t        = pred_per_frame[t]
            gt_remaining_t = transform_to_frame_t(gt_3[t-CUR_IDX:], ego_poses_all, t)
        else:
            pred_t        = transform_to_frame_t(pred_per_frame[CUR_IDX], ego_poses_all, t)
            gt_remaining_t = transform_to_frame_t(gt_3, ego_poses_all, t)

        draw_bev_traj(ax_bev, clip_traj(gt_remaining_t, polys), COLOR_GT,   lw=2.0)
        draw_bev_traj(ax_bev, clip_traj(pred_t, polys),         COLOR_PRED, lw=2.8)

        dot_color = '#4477AA' if t <= CUR_IDX else '#FF8800'
        ax_bev.plot(0, 0, 'o', color=dot_color, markersize=9, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)

        configure_bev_ax(ax_bev)
        ax_bev.set_xlim(30, -30); ax_bev.set_ylim(-5, 55)
        configure_ax(ax_bev); ax_bev.set_aspect('auto')
        for sp in ax_bev.spines.values(): sp.set_linewidth(1.2)

        camera = d['sc_cam'].frames[t].cameras.cam_f0
        img = camera.image
        if not isinstance(img, np.ndarray) or img.dtype == object:
            try:
                from PIL import Image as PI
                img = np.array(PI.open(str(img.flat[0])).convert('RGB'))
            except Exception:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img = prepare_cam_image(img.copy())
        draw_cam_traj(img, project_traj(clip_traj(gt_remaining_t, polys), camera), hex2bgr(COLOR_GT))
        draw_cam_traj(img, project_traj(clip_traj(pred_t, polys),          camera), hex2bgr(COLOR_PRED))

        ax_cam.imshow(img[:,:,::-1]); ax_cam.set_aspect('auto')
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        for sp in ax_cam.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.2)

        bbox_bev = ax_bev.get_position(); bbox_cam = ax_cam.get_position()
        x_c = (bbox_bev.x0 + bbox_cam.x1) / 2
        y_b = min(bbox_bev.y0, bbox_cam.y0)
        fig.text(x_c, y_b - 0.012, label, fontsize=22, fontweight='bold',
                 ha='center', va='top')


    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return img_frame

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tokens = [tok for tok, _ in SCENES]

    # ── Phase 1: 8-GPU parallel inference ────────────────────────────────────
    all_tasks = [(tok, t) for tok in tokens for t in range(CUR_IDX, N_FRAMES)]
    n_gpu = min(8, torch.cuda.device_count())
    print(f'Phase 1: {len(all_tasks)} inference tasks on {n_gpu} GPUs ...')
    mp.spawn(inference_worker, args=(all_tasks,), nprocs=n_gpu, join=True)
    print('Phase 1 done.')

    # ── Load BEV/camera data & predictions ───────────────────────────────────
    loader_bev = _make_loader(tokens, SensorConfig.build_no_sensors(), 'bev')
    loader_cam = _make_loader(tokens, SensorConfig(
        cam_f0=list(range(N_FRAMES)),
        cam_l0=[], cam_l1=[], cam_l2=[],
        cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
        lidar_pc=[],
    ), 'cam')

    scene_data = {}
    for token, label in SCENES:
        sc_bev = loader_bev.get_scene_from_token(token)
        sc_cam = loader_cam.get_scene_from_token(token)
        hist   = sc_bev.get_history_trajectory().poses
        fut    = sc_bev.get_future_trajectory().poses
        ego_poses_all = np.zeros((N_FRAMES, 3), np.float32)
        for i in range(len(hist)): ego_poses_all[i] = hist[i]
        for i in range(len(fut)):  ego_poses_all[CUR_IDX+1+i] = fut[i]
        gt_3 = sc_bev.get_future_trajectory().poses

        pred_per_frame = {}
        for t in range(CUR_IDX, N_FRAMES):
            with open(TMP_DIR / f'{token}_{t}.pkl', 'rb') as f:
                pred_per_frame[t] = pickle.load(f)

        scene_data[token] = dict(sc_bev=sc_bev, sc_cam=sc_cam,
                                  gt_3=gt_3, ego_poses_all=ego_poses_all,
                                  pred_per_frame=pred_per_frame)
        print(f'  {label}: loaded')

    # ── Phase 2: render ───────────────────────────────────────────────────────
    print(f'Phase 2: rendering {N_FRAMES} frames ...')
    frames = []
    for t in range(N_FRAMES):
        frames.append(render_frame(t, scene_data))
        print(f'  [{t+1}/{N_FRAMES}]')

    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    out_gif = OUT_DIR / 'demo_4scenes_v3.gif'
    pil_frames[0].save(str(out_gif), save_all=True, append_images=pil_frames[1:],
                       duration=500, loop=0, optimize=False)
    print(f'\nGIF saved: {out_gif}  ({out_gif.stat().st_size//1024} KB)')
