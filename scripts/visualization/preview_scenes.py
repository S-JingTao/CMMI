"""
Quick BEV preview of multiple scene tokens (no model inference).
Generates a grid image of camera + BEV for N tokens to find straight-road scenes.
"""
import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['LOCAL_RANK']          = '0'
os.environ['OPENSCENE_DATA_ROOT'] = '/path/to/dataset'
os.environ['NUPLAN_MAPS_ROOT']    = '/path/to/dataset/maps'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/path/to/dataset')
OUT_DIR   = ROOT / 'exp/ablation_vis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax

# Candidate tokens (perfect score + lead vehicle)
TOKENS = [
    '001b34b45b2f50e4',
    '0027db600afe56ca',
    '008d66a74c275479',
    '008d6e3394a65c1f',
    '012ca60989175c54',
    '012dc5d8043555ef',
    '013456829c0050ce',
    '013fbdcd9db35b43',
    '0165d0814e905c1c',
    '0179d579d30e588c',
    '0192c3bca9ca5c67',
    '01a58976a2e45a3d',
    '01c4dab17c975e13',
    '022cc20c8dd45bc5',
    '026bb114391d5b81',
    '026ee9bc920b5180',
    '028d10ed5c105755',
    '02e1537a43d55ab2',
    '02e3c48291855ae8',
    '03388b830f975734',
]

def load_scene(token):
    GlobalHydra.instance().clear()
    cfg_dir = str(ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None, job_name='preview'):
        sf = instantiate(hydra.compose(config_name='navtest'))
    sf.tokens = [token]
    loader = SceneLoader(
        sensor_blobs_path=DATA_ROOT / 'sensor_blobs/test',
        data_path=DATA_ROOT / 'navsim_logs/test',
        scene_filter=sf,
        sensor_config=SensorConfig(
            cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[],
            cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
            lidar_pc=[0],
        ),
    )
    return loader.get_scene_from_token(token)


def main():
    n = len(TOKENS)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 4, rows * 4))
    axes = axes.reshape(rows, cols * 2)

    for idx, token in enumerate(TOKENS):
        r = idx // cols
        c = (idx % cols) * 2
        print(f'[{idx+1}/{n}] {token}')

        try:
            scene = load_scene(token)
            cur = scene.frames[scene.scene_metadata.num_history_frames - 1]

            # Camera
            cam = cur.cameras.cam_f0.image
            if cam.dtype == object:
                cam = np.array(PILImage.open(str(cam.flat[0])).convert('RGB'))
            ax_cam = axes[r, c]
            ax_cam.imshow(cam.astype(np.uint8))
            ax_cam.axis('off')
            ax_cam.set_title(token[:8], fontsize=7)

            # BEV
            ax_bev = axes[r, c + 1]
            add_configured_bev_on_ax(ax_bev, scene.map_api, cur)
            gt = scene.get_future_trajectory().poses[:, :2]
            ax_bev.plot(gt[:, 1], gt[:, 0], 'g-', lw=2)
            ax_bev.set_xlim(12, -12)
            ax_bev.set_ylim(-2, 28)
            ax_bev.axis('off')
        except Exception as e:
            axes[r, c].set_title(f'{token[:8]}\nERR', fontsize=7)
            axes[r, c].axis('off')
            axes[r, c + 1].axis('off')
            print(f'  ERROR: {e}')

    # Hide unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = (idx % cols) * 2
        axes[r, c].axis('off')
        axes[r, c + 1].axis('off')

    out = OUT_DIR / 'scene_preview.png'
    fig.savefig(str(out), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
