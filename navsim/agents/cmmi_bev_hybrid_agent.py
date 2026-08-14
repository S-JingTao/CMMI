"""CMMIBevHybridAgent: BEV-token front-end + Hybrid v3 back-end.

Weight loading order:
  1. from_pretrained(cmmi_path)        → LLM (BEV-token fine-tuned)
  2. add_bev_encoder(bev_ckpt_path)        → BEVFusion backbone (frozen)
  3. cmmi_path/model.safetensors       → bev_encoder.*
                                              (BEVProjector trained with BEV token objective,
                                               plus initial seg/det heads — overridden in step 4)
  4. seg_det_ckpt_path → bev_encoder.seg_head.* + bev_encoder.det_head.* ONLY
                          (Phase-1 GT-supervised heads; BEVProjector is NOT touched)
"""

import os
from safetensors.torch import load_file as _sf_load

from navsim.agents.cmmi_bev_agent import CMMIBevAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class CMMIBevHybridAgent(CMMIBevAgent):
    """BEV-token-injected front-end (our fine-tuned LLM + BEVProjector)
    combined with Hybrid v3's Phase-1 GT-supervised seg/det back-end.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        fudoki_path: str = "",
        cmmi_path: str = "",
        bev_ckpt_path: str = "",
        seg_det_ckpt_path: str = "",
        text_embedding_path: str = "",
        image_embedding_path: str = "",
        heading_mlp_path: str = "",
        discrete_fm_steps: int = 5,
        seed: int = 99,
        dtype: str = "default",
        bev_cache_dir: str = "",
    ):
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            fudoki_path=fudoki_path,
            cmmi_path=cmmi_path,
            bev_ckpt_path=bev_ckpt_path,
            text_embedding_path=text_embedding_path,
            image_embedding_path=image_embedding_path,
            heading_mlp_path=heading_mlp_path,
            discrete_fm_steps=discrete_fm_steps,
            seed=seed,
            dtype=dtype,
            bev_cache_dir=bev_cache_dir,
        )
        self.seg_det_ckpt_path = seg_det_ckpt_path

    def initialize(self) -> None:
        # Steps 1-3: LLM (BEV-token fine-tuned) + BEVFusion backbone + BEVProjector
        super().initialize()

        # Step 4: override ONLY seg_head + det_head with Phase-1 GT-supervised weights.
        # BEVProjector keys are intentionally excluded so the front-end BEV injection is preserved.
        if self.seg_det_ckpt_path and os.path.exists(self.seg_det_ckpt_path):
            sd = _sf_load(self.seg_det_ckpt_path, device='cpu')
            seg_det_sd = {
                k[len('bev_encoder.'):]: v
                for k, v in sd.items()
                if k.startswith('bev_encoder.seg_head.') or k.startswith('bev_encoder.det_head.')
            }
            model_bev_sd = self.model.bev_encoder.state_dict()
            filtered = {
                k: v for k, v in seg_det_sd.items()
                if k in model_bev_sd and v.shape == model_bev_sd[k].shape
            }
            miss, unexp = self.model.bev_encoder.load_state_dict(filtered, strict=False)
            print(f'[CMMIBevHybridAgent] Phase-1 seg/det loaded: '
                  f'matched={len(filtered)}, missing={len(miss)}, unexpected={len(unexp)}')
        else:
            print('[CMMIBevHybridAgent] WARNING: no seg_det_ckpt_path, '
                  'using seg/det heads from training checkpoint')
