"""Run CMMIHybridAgent (newly tuned, cmmi_hybrid_eval checkpoint config) on 3 specific tokens."""
import os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

DATA_ROOT = Path(os.environ.get('OPENSCENE_DATA_ROOT', 'data/dataset'))
REPO_ROOT = Path(__file__).parent
OUT_DIR   = Path('output/infer/hybrid')

TOKENS = [
    'b8104b69b2d9509b',
]

os.environ['OPENSCENE_DATA_ROOT'] = str(DATA_ROOT)

from navsim.agents.cmmi_hybrid_agent import CMMIHybridAgent
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.common.dataloader import SceneLoader

agent = CMMIHybridAgent(
    fudoki_path=str(REPO_ROOT / 'pretrained_model/fudoki'),
    cmmi_path=str(REPO_ROOT / 'pretrained_model/cmmi'),
    bev_ckpt_path=str(REPO_ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth'),
    seg_det_ckpt_path=str(REPO_ROOT / 'output/pretrain/seg_det/checkpoint-20000/model.safetensors'),
    text_embedding_path=str(REPO_ROOT / 'pretrained_model/fudoki/text_embedding.pt'),
    image_embedding_path=str(REPO_ROOT / 'pretrained_model/fudoki/image_embedding.pt'),
    heading_mlp_path=str(REPO_ROOT / 'pretrained_model/cmmi/best_model_epoch95.pt'),
    discrete_fm_steps=5,
)
agent.initialize()

GlobalHydra.instance().clear()
config_dir = str(REPO_ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
    scene_filter = instantiate(hydra.compose(config_name='navtest'))

scene_loader = SceneLoader(
    sensor_blobs_path=DATA_ROOT / 'sensor_blobs/test',
    data_path=DATA_ROOT / 'navsim_logs/test',
    scene_filter=scene_filter,
    sensor_config=agent.get_sensor_config(),
    load_image_path=True,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for token in TOKENS:
    out_path = OUT_DIR / f'{token}.npy'
    agent_input = scene_loader.get_agent_input_from_token(token)
    scene = scene_loader.get_scene_from_token(token)
    traj = agent.compute_trajectory(agent_input, scene)
    np.save(out_path, traj.poses.astype(np.float32))
    print(f'{token}: saved {traj.poses.shape} -> {out_path}')

print('Done.')
