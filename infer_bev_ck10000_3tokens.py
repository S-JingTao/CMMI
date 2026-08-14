"""Run CMMIBevAgent (Section 3.2, checkpoint-10000, directly trained) on 3 specific tokens."""
import os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

DATA_ROOT = Path(os.environ.get('OPENSCENE_DATA_ROOT', 'data/dataset'))
REPO_ROOT = Path(__file__).parent
OUT_DIR   = Path('output/infer/bev')

TOKENS = [
    '99ebd32df2f453f8',
    'c38eb1a738745b1e',
    'a41f538fa8e25be0',
]

os.environ['OPENSCENE_DATA_ROOT'] = str(DATA_ROOT)

import torch.nn as nn
_orig_load_state_dict = nn.Module.load_state_dict
def _shape_filtered_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    """checkpoint-10000 predates the Phase-1 det_head architecture change
    (128ch/10cls -> 64ch/1cls) — silently skip mismatched-shape keys instead
    of erroring, matching the shape-filtered strict=False pattern used
    elsewhere in this codebase for the same reason."""
    own_sd = self.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in own_sd and v.shape == own_sd[k].shape}
    return _orig_load_state_dict(self, filtered, strict=False)
nn.Module.load_state_dict = _shape_filtered_load_state_dict

from navsim.agents.cmmi_bev_agent import CMMIBevAgent
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.common.dataloader import SceneLoader

agent = CMMIBevAgent(
    fudoki_path=str(REPO_ROOT / 'pretrained_model/fudoki'),
    cmmi_path=str(REPO_ROOT / 'output/train/navsim_bev/checkpoint-10000'),
    bev_ckpt_path=str(REPO_ROOT / 'pretrained_model/bevfusion/bevfusion_lidar_nusc.pth'),
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
