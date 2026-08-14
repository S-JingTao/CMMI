"""
Visualize CMMI BEV agent trajectories on high-scoring navtest scenes.
Usage:
  conda run -n cmmi python scripts/visualization/visualize_cmmi_bev.py
"""
import os
import sys
import random
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUPLAN_MAPS_ROOT",    "/path/to/dataset/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/path/to/dataset")

EVAL_CSV   = str(ROOT / "exp/cmmi_bev_agent_eval/2026.06.12.15.14.29/2026.06.12.16.25.56.csv")
CKPT       = str(ROOT / "output/train/navsim_bev/checkpoint-40000")
OUT_DIR    = ROOT / "exp/cmmi_bev_vis"
NUM_SCENES = 20          # how many scenes to visualize
SCORE_MIN  = 0.95        # only pick scenes above this score
SPLIT      = "test"   # navtest uses data_split=test
SEED       = 42

# ── agent config ───────────────────────────────────────────────────────────
FUDOKI_PATH  = str(ROOT / "pretrained_model/fudoki")
BEV_CKPT     = str(ROOT / "pretrained_model/bevfusion/bevfusion_lidar_nusc.pth")
TEXT_EMB     = str(ROOT / "pretrained_model/fudoki/text_embedding.pt")
IMAGE_EMB    = str(ROOT / "pretrained_model/fudoki/image_embedding.pt")
HEADING_MLP  = str(ROOT / "pretrained_model/cmmi/best_model_epoch95.pt")
FM_STEPS     = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. pick high-score tokens (intersect with scene_loader after it's ready)
    df = pd.read_csv(EVAL_CSV)
    high = set(df[df["score"] >= SCORE_MIN]["token"].tolist())
    print(f"High-score tokens (>={SCORE_MIN}): {len(high)}")

    # 2. load scene loader
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from hydra import compose, initialize_config_dir

    sensor_cfg = SensorConfig(
        cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[],
        cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
        lidar_pc=[3],
    )

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])

    # Use navtest scene filter to match evaluation conditions
    cfg_path = str(ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter")
    with hydra.initialize_config_dir(config_dir=cfg_path, job_name="vis"):
        filter_cfg = hydra.compose(config_name="navtest")
    from hydra.utils import instantiate
    scene_filter = instantiate(filter_cfg)

    scene_loader = SceneLoader(
        sensor_blobs_path=data_root / f"sensor_blobs/{SPLIT}",
        data_path=data_root / f"navsim_logs/{SPLIT}",
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )
    print(f"SceneLoader ready, {len(scene_loader.tokens)} tokens available")

    # intersect high-score tokens with those in scene_loader
    available = list(high & set(scene_loader.tokens))
    print(f"High-score tokens in scene_loader: {len(available)}")
    random.seed(SEED)
    selected = random.sample(available, min(NUM_SCENES, len(available)))
    print(f"Selected {len(selected)} tokens for visualization")

    # 3. init agent
    import torch
    from navsim.agents.cmmi_bev_agent import CMMIBevAgent

    os.environ["LOCAL_RANK"] = "0"
    agent = CMMIBevAgent(
        fudoki_path=FUDOKI_PATH,
        cmmi_path=CKPT,
        bev_ckpt_path=BEV_CKPT,
        text_embedding_path=TEXT_EMB,
        image_embedding_path=IMAGE_EMB,
        heading_mlp_path=HEADING_MLP,
        discrete_fm_steps=FM_STEPS,
    )
    agent.initialize()
    print("Agent initialized")

    # 4. visualize
    from navsim.visualization.plots import plot_bev_and_camera_with_agent

    saved = 0
    for token in selected:
        try:
            scene = scene_loader.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            score_row = df[df["token"] == token]["score"].values
            score = float(score_row[0]) if len(score_row) else -1.0

            fig, bev_ax, cam_ax = plot_bev_and_camera_with_agent(
                scene, scene, frame_idx, agent
            )
            fig.suptitle(f"token={token}  score={score:.4f}", fontsize=10)
            out_path = OUT_DIR / f"{token}_score{score:.3f}.png"
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved: {out_path.name}")
            saved += 1
        except Exception as e:
            print(f"  ERROR {token}: {e}")

    print(f"\nDone — {saved}/{len(selected)} images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
