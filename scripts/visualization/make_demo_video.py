"""
Generate a demo video for a single navtest scene using CMMIHybridAgent.

Layout per frame:
  left : BEV bird's-eye view (map + agent trajectory + GT trajectory)
  right: front camera with projected trajectories

All scene frames (history + future) are rendered; the agent runs once at the
current frame and the predicted trajectory is frozen across all frames.

Usage:
  
  conda run -n cmmi python scripts/visualization/make_demo_video.py \
      [--token <TOKEN>] [--out_dir <DIR>] [--fps 4]
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENSCENE_DATA_ROOT", "/path/to/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT",    "/path/to/dataset/maps")

# ── config ──────────────────────────────────────────────────────────────────
EVAL_CSV   = str(ROOT / "exp/baseline_hybrid_eval/2026.07.06.14.41.29/2026.07.06.15.56.37.csv")
OUT_DIR    = ROOT / "exp/demo_video"
FPS        = 4

FUDOKI_PATH    = str(ROOT / "pretrained_model/fudoki")
CMMI_PATH  = str(ROOT / "pretrained_model/cmmi")
BEV_CKPT       = str(ROOT / "pretrained_model/bevfusion/bevfusion_lidar_nusc.pth")
SEG_DET_CKPT   = str(ROOT / "output/pretrain/seg_det/checkpoint-20000/model.safetensors")
TEXT_EMB       = str(ROOT / "pretrained_model/fudoki/text_embedding.pt")
IMAGE_EMB      = str(ROOT / "pretrained_model/fudoki/image_embedding.pt")
HEADING_MLP    = str(ROOT / "pretrained_model/cmmi/best_model_epoch95.pt")
FM_STEPS       = 5

TRAJ_AGENT = dict(
    line_color="#FF4444", line_color_alpha=0.9, line_width=2.5, line_style="-",
    marker="o", marker_size=6, marker_edge_color="#CC0000",
    fill_color="#FF4444", fill_color_alpha=0.2,
    zorder=10,
)
TRAJ_HUMAN = dict(
    line_color="#44AA44", line_color_alpha=0.85, line_width=2.0, line_style="--",
    marker="o", marker_size=5, marker_edge_color="#226622",
    fill_color="#44AA44", fill_color_alpha=0.15,
    zorder=9,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token",   type=str, default=None, help="scene token; default: best score in CSV")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--fps",      type=int,   default=20)
    p.add_argument("--duration", type=float, default=3.0, help="output video length in seconds")
    return p.parse_args()


def load_scene(token):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    sensor_cfg = SensorConfig(
        cam_f0=list(range(14)),   # load all 14 frames
        cam_l0=[], cam_l1=[], cam_l2=[],
        cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
        lidar_pc=[3],
    )
    # navtest uses data_split=test with specific log_names (loaded via navtest scene filter)
    import hydra
    cfg_path = str(ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter")
    with hydra.initialize_config_dir(config_dir=cfg_path, job_name="demo"):
        filter_cfg = hydra.compose(config_name="navtest")
    from hydra.utils import instantiate
    scene_filter = instantiate(filter_cfg)
    scene_filter.tokens = [token]

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    loader = SceneLoader(
        sensor_blobs_path=data_root / "sensor_blobs/test",
        data_path=data_root / "navsim_logs/test",
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )
    assert token in loader.tokens, f"Token {token} not found in navtest"
    return loader.get_scene_from_token(token)


def init_agent():
    os.environ["LOCAL_RANK"] = "0"
    from navsim.agents.cmmi_hybrid_agent import CMMIHybridAgent
    agent = CMMIHybridAgent(
        fudoki_path=FUDOKI_PATH,
        cmmi_path=CMMI_PATH,
        bev_ckpt_path=BEV_CKPT,
        seg_det_ckpt_path=SEG_DET_CKPT,
        text_embedding_path=TEXT_EMB,
        image_embedding_path=IMAGE_EMB,
        heading_mlp_path=HEADING_MLP,
        discrete_fm_steps=FM_STEPS,
    )
    agent.initialize()
    return agent


def render_frame(scene, frame_idx, agent_traj, human_traj, cur_idx, score, out_path):
    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.camera import add_camera_ax, add_trajectory_to_camera_ax

    frame = scene.frames[frame_idx]
    is_current = (frame_idx == cur_idx)
    is_future  = (frame_idx > cur_idx)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    bev_ax, cam_ax = axes

    # ── BEV ──────────────────────────────────────────────────────────────────
    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)

    # GT trajectory (dashed green)
    add_trajectory_to_bev_ax(bev_ax, human_traj, TRAJ_HUMAN)

    # Agent trajectory (red solid) — shown from current frame onward
    if frame_idx >= cur_idx:
        add_trajectory_to_bev_ax(bev_ax, agent_traj, TRAJ_AGENT)

    # Ego position marker
    bev_ax.plot(0, 0, "k^", markersize=10, zorder=10, label="Ego")

    bev_ax.set_title(f"BEV  frame {frame_idx}/{len(scene.frames)-1}", fontsize=11)
    bev_ax.legend(fontsize=8, loc="upper right")

    # ── Camera ───────────────────────────────────────────────────────────────
    cam = frame.cameras.cam_f0
    # ensure image is a uint8 numpy array (some frames may store path strings)
    if not isinstance(cam.image, np.ndarray) or cam.image.dtype == object:
        from PIL import Image as PILImage
        img_arr = np.array(PILImage.open(str(cam.image.flat[0])).convert("RGB"))
        import dataclasses
        cam = dataclasses.replace(cam, image=img_arr)
    add_camera_ax(cam_ax, cam)

    # Project trajectories onto camera at or after current frame
    if frame_idx >= cur_idx:
        add_trajectory_to_camera_ax(cam_ax, cam, agent_traj, {
            "line_color": "#FF4444", "line_color_alpha": 0.9,
            "line_width": 2.0, "line_style": "-", "zorder": 10,
            "arrow_color": "#FF4444", "arrow_edge_color": "#CC0000",
            "arrow_alpha": 0.9, "arrow_line_width": 1.5,
        })
        add_trajectory_to_camera_ax(cam_ax, cam, human_traj, {
            "line_color": "#44AA44", "line_color_alpha": 0.8,
            "line_width": 1.5, "line_style": "--", "zorder": 9,
            "arrow_color": "#44AA44", "arrow_edge_color": "#226622",
            "arrow_alpha": 0.8, "arrow_line_width": 1.2,
        })

    # Frame label
    frame_label = "CURRENT" if is_current else ("FUTURE" if is_future else "HISTORY")
    cam_ax.set_title(f"Front Camera  [{frame_label}]  (score={score:.4f})", fontsize=11)

    fig.suptitle(f"CMMI Hybrid Demo  |  token={scene.scene_metadata.log_name}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def frames_to_video(frames_dir: Path, out_path: Path, fps: int,
                    duration_sec: float = 3.0):
    """Interpolate source frames to `duration_sec` seconds at `fps` using linear blending."""
    import cv2

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    assert frame_files, f"No frames found in {frames_dir}"

    # load all source frames once
    src = [cv2.imread(str(f)).astype(np.float32) for f in frame_files]
    n_src = len(src)                        # e.g. 14
    n_out = round(duration_sec * fps)       # e.g. 3 * 20 = 60
    h, w = src[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for j in range(n_out):
        # map output frame j → position in source frame sequence
        pos = j * (n_src - 1) / (n_out - 1)
        lo  = int(pos)
        hi  = min(lo + 1, n_src - 1)
        alpha = pos - lo
        interp = ((1.0 - alpha) * src[lo] + alpha * src[hi]).clip(0, 255).astype(np.uint8)
        vw.write(interp)

    vw.release()
    sz = out_path.stat().st_size // 1024
    print(f"Video: {out_path}  {sz} KB  "
          f"({w}x{h} @ {fps}fps  {n_src} src→{n_out} frames  {duration_sec:.1f}s)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── pick token ────────────────────────────────────────────────────────────
    df = pd.read_csv(EVAL_CSV)
    if args.token:
        token = args.token
        score_rows = df[df["token"] == token]["score"].values
        score = float(score_rows[0]) if len(score_rows) else -1.0
    else:
        best = df.sort_values("score", ascending=False).iloc[0]
        token, score = best["token"], float(best["score"])
    print(f"Token: {token}  score: {score:.4f}")

    # ── load scene ────────────────────────────────────────────────────────────
    print("Loading scene...")
    scene = load_scene(token)
    num_frames = len(scene.frames)
    cur_idx    = scene.scene_metadata.num_history_frames - 1
    print(f"  frames={num_frames}  current_idx={cur_idx}")

    # ── init agent & compute trajectory once ─────────────────────────────────
    print("Initialising agent...")
    agent = init_agent()

    print("Computing trajectory...")
    agent_input = scene.get_agent_input()
    agent_traj  = agent.compute_trajectory(agent_input)
    human_traj  = scene.get_future_trajectory()
    print(f"  agent_traj poses shape: {agent_traj.poses.shape}")

    # ── render frames ─────────────────────────────────────────────────────────
    print(f"Rendering {num_frames} frames...")
    for i in range(num_frames):
        out_path = frames_dir / f"frame_{i:03d}.png"
        render_frame(scene, i, agent_traj, human_traj, cur_idx, score, out_path)
        print(f"  [{i+1}/{num_frames}] {out_path.name}")

    # ── stitch video ──────────────────────────────────────────────────────────
    video_path = out_dir / f"demo_{token}.mp4"
    frames_to_video(frames_dir, video_path, args.fps, args.duration)
    print(f"\nVideo saved: {video_path}")


if __name__ == "__main__":
    main()
