"""CMMIHybridAgent: baseline front-end (no BEV token in LLM) +
our K-trajectory DFM + GPU-LM optimisation back-end (Section 3.2 / 3.3).

BEV encoder still runs for seg/det perception used in Γ_risk, but its
bev_tokens are NOT injected into the LLM sequence.
"""

import os
from typing import Optional

import numpy as np
import torch

from safetensors.torch import load_file as _sf_load

from navsim.agents.cmmi_bev_agent import (
    CMMIBevAgent,
    IMG_LEN,
    fmt_signed_2dec,
    map_command_to_direction,
    get_dtype,
)
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor, MultiModalityCausalLM
from fudoki.janus.models.heading_mlp import TrajectoryHeadingMLP
from torchvision import transforms
from flow_matching.data.navsim import resize_pad


class CMMIHybridAgent(CMMIBevAgent):
    """Front-end = baseline (image-only LLM, no BEV token injection).
    Back-end  = K-traj DFM + GPU LM optimisation with BEV seg/det perception.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        fudoki_path: str = "",
        cmmi_path: str = "",        # should point to pretrained_model/cmmi (original)
        bev_ckpt_path: str = "",
        seg_det_ckpt_path: str = "",    # Phase-1 pretrained seg/det weights
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
        self.txt_max_length = 500   # cmmi: no BEV tokens in sequence

    # ── model loading ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        import torch.backends.cudnn as cudnn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cudnn.benchmark = True

        self.transform_img = transforms.Compose([
            transforms.Lambda(resize_pad),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.fudoki_path)

        interval = int((self.quantize_max_num - self.quantize_min_num) / self.quantize_interval) + 1
        num_tokens = [f"{x:.2f}" for x in np.linspace(self.quantize_min_num, self.quantize_max_num, interval)]
        self.vl_chat_processor.tokenizer.add_tokens(num_tokens)

        # ── Load baseline model (no BEV token fine-tuning) ──────────
        model = MultiModalityCausalLM.from_pretrained(self.cmmi_path).to(self.device, dtype=self.dtype)
        model.eval()
        model.train(False)

        # ── Attach BEV encoder (perception-only, NOT injected into LLM) ─────
        model.add_bev_encoder(self.bev_ckpt_path)

        # Load Phase-1 pretrained seg/det weights (shape-filtered, strict=False)
        if self.seg_det_ckpt_path and os.path.exists(self.seg_det_ckpt_path):
            sd = _sf_load(self.seg_det_ckpt_path, device='cpu')
            bev_sd = {k[len('bev_encoder.'):]: v for k, v in sd.items() if k.startswith('bev_encoder.')}
            model_bev_sd = model.bev_encoder.state_dict()
            filtered = {k: v for k, v in bev_sd.items()
                        if k in model_bev_sd and v.shape == model_bev_sd[k].shape}
            miss, unexp = model.bev_encoder.load_state_dict(filtered, strict=False)
            print(f'[CMMIHybridAgent] seg/det weights loaded: '
                  f'matched={len(filtered)}, missing={len(miss)}, unexpected={len(unexp)}')
        else:
            print('[CMMIHybridAgent] WARNING: no seg_det_ckpt_path, seg/det heads are random')

        model.bev_encoder = model.bev_encoder.to(self.device)
        model.bev_encoder.eval()
        self.model = model

        cfg_model = CFGScaledModel(model, g_or_u='understanding')

        path_txt = MixtureDiscreteSoftmaxProbPath(mode='text', embedding_path=self.text_embedding_path)
        path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=self.image_embedding_path)

        self.vocabulary_size_txt = max(
            len(self.vl_chat_processor.tokenizer),
            model.language_model.get_input_embeddings().weight.shape[0],
        )
        with torch.no_grad():
            path_txt.set_embedding(model.language_model.get_input_embeddings())
            self.solver = MixtureDiscreteSoftmaxEulerSolver(
                model=cfg_model,
                path_txt=path_txt,
                path_img=path_img,
                vocabulary_size_txt=self.vocabulary_size_txt,
                vocabulary_size_img=model.config.gen_vision_config.params.image_token_size,
            )

        heading_model = TrajectoryHeadingMLP(hidden_dims=[512, 512, 256, 128])
        state_dict = torch.load(self.heading_mlp_path, map_location=self.device)
        heading_model.load_state_dict(state_dict)
        heading_model.eval()
        self.heading_model = heading_model.to(self.device)

    # ── input preparation (image-only, no BEV token in LLM sequence) ────────

    def _build_solver_inputs(self, agent_input: AgentInput, scene=None):
        from PIL import Image as PILImage

        if scene is not None:
            navigation_info = self.get_current_navigation_infomation(scene)
            state = self.get_ego_status(scene)
        else:
            navigation_info, state = self._nav_state_from_agent_input(agent_input)

        raw_img = agent_input.cameras[-1].cam_f0.image
        if isinstance(raw_img, np.ndarray):
            img = PILImage.fromarray(raw_img.astype(np.uint8)).convert("RGB")
        else:
            img = PILImage.open(str(raw_img)).convert("RGB")

        lidar_frame = agent_input.lidars[-1]
        has_bev = lidar_frame is not None and lidar_frame.lidar_pc is not None
        bev_points = lidar_frame.lidar_pc.T if has_bev else None  # [N,6]

        init_pos_str = "(0.00,0.00)"
        # ── Baseline prompt: no <bev_placeholder> ──────────────────
        conversation = [
            {
                "role": "User",
                "content": (
                    "Here is front views of a driving vehicle:\n"
                    "<image_placeholder>\n"
                    f"The navigation information is: {navigation_info}\n"
                    f"The current position is {init_pos_str}\n"
                    f"Current velocity is: {state[0]}  and current accelerate is: {state[1]}\n"
                    "Predict the optimal driving action for the next 4 seconds with 8 new waypoints."
                ),
            },
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )

        img_tensor = self.transform_img(img)
        input_ids = torch.LongTensor(self.vl_chat_processor.tokenizer.encode(sft_format))

        image_token_mask = (input_ids == self.vl_chat_processor.image_id)
        input_ids, _ = self.vl_chat_processor.add_image_token(
            image_indices=image_token_mask.nonzero(),
            input_ids=input_ids,
        )

        # No BEV token expansion; mask is all-zeros so modeling_vlm skips injection
        bev_expanded_token_mask = torch.zeros_like(input_ids)

        total_special = IMG_LEN   # 576 image tokens only
        rows_to_pad = max(self.txt_max_length + total_special - input_ids.shape[0], 100)
        input_ids = torch.cat(
            [input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0
        )
        attention_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)

        image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
        input_ids[torch.where(image_expanded_token_mask == 1)[0]] = 0

        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = self.vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_token_length = len(split_token)
        start_index = -1
        for j in range(len(input_ids) - split_token_length + 1):
            if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
                start_index = j
                break
        if start_index == -1:
            raise ValueError("Split token not found in input_ids")
        text_expanded_token_mask[(start_index + split_token_length):] = 1

        data_info = {}
        data_info['text_token_mask']  = text_expanded_token_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        # all-zeros → modeling_vlm bev_mask.any() == False → no BEV injection
        data_info['bev_token_mask']   = bev_expanded_token_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        data_info['generation_or_understanding_mask'] = (
            torch.zeros(self.batch_size, 1, dtype=torch.int, device=self.device)
        )
        data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        data_info['sft_format'] = sft_format
        data_info['understanding_img'] = img_tensor.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(self.device)
        data_info['has_understanding_img'] = torch.ones(self.batch_size, 1, dtype=torch.int, device=self.device)
        data_info['has_bev'] = (
            torch.full((self.batch_size, 1), 1 if has_bev else 0, dtype=torch.int, device=self.device)
        )
        data_info['bev_points'] = [bev_points] * self.batch_size   # kept for _get_bev_perception

        input_ids = input_ids.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        x_0_txt = torch.randint(self.vocabulary_size_txt, input_ids.shape, dtype=torch.long, device=self.device)
        x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])

        return x_init, data_info
