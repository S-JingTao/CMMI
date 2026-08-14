"""
Quick test: verify compute_trajectory_multi() returns K valid trajectories.

Usage:
  
  conda run -n cmmi python scripts/test_multi_traj.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUPLAN_MAPS_ROOT",    "/path/to/dataset/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/path/to/dataset")
os.environ["LOCAL_RANK"] = "0"

import numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.cmmi_bev_agent import CMMIBevAgent

# ── config ──────────────────────────────────────────────────────────────
CKPT        = str(ROOT / "output/train/navsim_bev/checkpoint-40000")
FUDOKI      = str(ROOT / "pretrained_model/fudoki")
BEV_CKPT    = str(ROOT / "pretrained_model/bevfusion/bevfusion_lidar_nusc.pth")
TEXT_EMB    = str(ROOT / "pretrained_model/fudoki/text_embedding.pt")
IMAGE_EMB   = str(ROOT / "pretrained_model/fudoki/image_embedding.pt")
HEADING_MLP = str(ROOT / "pretrained_model/cmmi/best_model_epoch95.pt")
FM_STEPS    = 5   # K = 5 → expect 5 candidate trajectories

# ── load one scene ───────────────────────────────────────────────────────
data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
scene_loader = SceneLoader(
    sensor_blobs_path=data_root / "sensor_blobs/test",
    data_path=data_root / "navsim_logs/test",
    scene_filter=SceneFilter(max_scenes=5),
    sensor_config=SensorConfig(
        cam_f0=[3], lidar_pc=[3],
        cam_l0=[], cam_l1=[], cam_l2=[],
        cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
    ),
)
token = scene_loader.tokens[0]
scene = scene_loader.get_scene_from_token(token)
frame_idx = scene.scene_metadata.num_history_frames - 1
agent_input = scene_loader.get_agent_input_from_token(token)

print(f"Token: {token}")

# ── init agent ───────────────────────────────────────────────────────────
agent = CMMIBevAgent(
    fudoki_path=FUDOKI,
    cmmi_path=CKPT,
    bev_ckpt_path=BEV_CKPT,
    text_embedding_path=TEXT_EMB,
    image_embedding_path=IMAGE_EMB,
    heading_mlp_path=HEADING_MLP,
    discrete_fm_steps=FM_STEPS,
)
agent.initialize()
print("Agent initialized")

# ── test single trajectory ────────────────────────────────────────────────
single = agent.compute_trajectory(agent_input)
print(f"\n[Single] poses shape: {single.poses.shape}")
print(f"  xy: {single.poses[:, :2]}")

# ── test multi trajectory ────────────────────────────────────────────────
candidates = agent.compute_trajectory_multi(agent_input)
print(f"\n[Multi]  {len(candidates)} candidates (K={FM_STEPS})")

for k, traj in enumerate(candidates, start=1):
    xy = traj.poses[:, :2]
    print(f"  τ̃({k})  x_range=[{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
          f"y_range=[{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")

# ── sanity: last candidate == single (same DFM run) ──────────────────────
diff = np.abs(candidates[-1].poses - single.poses).max()
print(f"\nMax diff between τ̃(K) and single trajectory: {diff:.4f}")
print("  (should be 0 — both are the final DFM step output)")
