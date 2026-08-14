import os
import re
from typing import Any, List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torchvision import transforms

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from flow_matching.data.navsim import resize_pad
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor, MultiModalityCausalLM
from fudoki.janus.models.heading_mlp import TrajectoryHeadingMLP

IMG_LEN = 576
BEV_LEN = 196

# ── BEV grid constants (must match BEV encoder config) ──
BEV_RANGE = 54.0    # metres, symmetric in ±x and ±y
BEV_RES   = 0.6     # metres per pixel  (2×54 / 180 = 0.6)
BEV_SIZE  = 180     # grid side length in pixels
# SEG_CLASSES = ['drivable_area','ped_crossing','walkway','lane_divider','vehicle','background']
# Paper Eq.(9): ω_c ≥ 0; drivable_area=0 (safe), others reflect danger
_SEG_WEIGHTS_RISK = np.array([0.0, 0.5, 1.0, 0.3, 2.0, 0.0], dtype=np.float32)
# Section 3.3 LM hyper-parameters
W_PRIOR     = 1.0   # Γ_prior weight (Eq.6)
W_SMOOTH_A  = 0.1   # acceleration smoothness weight (Eq.10)
W_SMOOTH_C  = 0.1   # curvature smoothness weight (Eq.10)
SIGMA_DET   = 1.0   # σ_j detection weight per bbox (Eq.9)
SIGMA_DECAY = 3.0   # σ distance decay in metres (Eq.7)


def fmt_signed_2dec(val: float) -> str:
    val = round(val, 2)
    if val == 0:
        val = 0.0
    return f"{val:.2f}"


def map_command_to_direction(command: np.ndarray) -> str:
    idx = np.argmax(command)
    if idx == 0:
        return "left"
    elif idx == 1:
        return "straight"
    elif idx == 2:
        return "right"
    return "unknown"


def get_dtype(dtype):
    if "fp16" in dtype:
        return torch.float16
    elif "bf16" in dtype:
        return torch.bfloat16
    else:
        return torch.float32


def resize_pad(image, image_size=384):
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))

    padding_color = (127, 127, 127)
    new_image = Image.new('RGB', (image_size, image_size), padding_color)

    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image


def extract_num_list(text):
    trajectory_numbers = []

    if "Trajectory:" in text and "Driving decision:" in text:
        trajectory_part = text.split("Trajectory:")[1]
        numbers_str = trajectory_part.split("Driving decision:")[0].strip()
        if re.match(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*$', numbers_str.replace(" ", "")):
            trajectory_numbers = [float(num.strip()) for num in numbers_str.split(",")]

    if not trajectory_numbers:
        pred_str = text.strip()
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', pred_str)
        trajectory_numbers = [float(n) for n in nums]

    if len(trajectory_numbers) < 16:
        last_valid_x = None
        last_valid_y = None

        if len(trajectory_numbers) % 2 == 0 and len(trajectory_numbers) >= 2:
            last_valid_x = trajectory_numbers[-2]
            last_valid_y = trajectory_numbers[-1]
        elif len(trajectory_numbers) % 2 == 1:
            last_valid_x = trajectory_numbers[-1]
            last_valid_y = trajectory_numbers[-2] if len(trajectory_numbers) >= 2 else last_valid_x

        need_complete = 16 - len(trajectory_numbers)

        if need_complete > 0 and last_valid_x is not None and last_valid_y is not None:
            complete_nums = []
            for i in range(need_complete // 2):
                complete_nums.append(last_valid_x)
                complete_nums.append(last_valid_y)
            if need_complete % 2 == 1:
                complete_nums.append(last_valid_x)
            trajectory_numbers += complete_nums[:need_complete]

    if len(trajectory_numbers) > 16:
        trajectory_numbers = trajectory_numbers[:16]
    elif len(trajectory_numbers) == 0:
        trajectory_numbers = [0.0] * 16

    return trajectory_numbers


class CMMIBevAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        fudoki_path: str = "",
        cmmi_path: str = "",
        bev_ckpt_path: str = "",
        text_embedding_path: str = "",
        image_embedding_path: str = "",
        heading_mlp_path: str = "",
        discrete_fm_steps: int = 50,
        seed: int = 99,
        dtype: str = "default",
        bev_cache_dir: str = "",
    ):
        super().__init__(trajectory_sampling)

        self.fudoki_path          = fudoki_path
        self.cmmi_path        = cmmi_path
        self.bev_ckpt_path        = bev_ckpt_path
        self.image_embedding_path = image_embedding_path
        self.text_embedding_path  = text_embedding_path
        self.heading_mlp_path     = heading_mlp_path

        self.discrete_fm_steps    = discrete_fm_steps
        self.bev_cache_dir        = bev_cache_dir

        self.batch_size           = 1
        self.txt_max_length       = 700   # match training config (baseline used 500)
        self.quantize_max_num     = 100
        self.quantize_min_num     = -100
        self.quantize_interval    = 0.01

        self.seed                 = seed
        self.dtype                = get_dtype(dtype)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = f"cuda:{local_rank}"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
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

        model = MultiModalityCausalLM.from_pretrained(self.cmmi_path).to(self.device, dtype=self.dtype)
        model.eval()
        model.train(False)

        # attach BEV encoder (backbone from bevfusion ckpt, projector from cmmi_path)
        model.add_bev_encoder(self.bev_ckpt_path)

        # load trained BEVProjector weights from checkpoint (from_pretrained ignores bev_encoder.* keys)
        import os as _os
        from safetensors.torch import load_file as _sf_load
        _ckpt_file = _os.path.join(self.cmmi_path, 'model.safetensors')
        if _os.path.exists(_ckpt_file):
            _ckpt_sd = _sf_load(_ckpt_file, device='cpu')
            _bev_sd = {k[len('bev_encoder.'):]: v for k, v in _ckpt_sd.items() if k.startswith('bev_encoder.')}
            if _bev_sd:
                # Filter by shape to avoid size-mismatch errors (e.g. det_head arch changes)
                _model_bev_sd = model.bev_encoder.state_dict()
                _bev_sd = {k: v for k, v in _bev_sd.items()
                           if k in _model_bev_sd and v.shape == _model_bev_sd[k].shape}
                _miss, _unexp = model.bev_encoder.load_state_dict(_bev_sd, strict=False)
                print(f'[CMMIBevAgent] BEVProjector loaded: missing={len(_miss)}, unexpected={len(_unexp)}')

        model.bev_encoder = model.bev_encoder.to(self.device)
        model.bev_encoder.eval()
        self.model = model   # keep reference for Section 3.3 BEV perception

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

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[3],   # current frame only, matches cam_f0
        )

    def get_current_navigation_infomation(self, scene) -> str:
        current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        navigation_command = np.array(current_frame.ego_status.driving_command)
        return map_command_to_direction(navigation_command)

    def get_ego_status(self, scene):
        start_frame_idx = scene.scene_metadata.num_history_frames - 1
        s = scene.frames[start_frame_idx].ego_status
        vel = s.ego_velocity.tolist()
        acc = s.ego_acceleration.tolist()
        vel_str = f"({fmt_signed_2dec(vel[0])},{fmt_signed_2dec(vel[1])})"
        acc_str = f"({fmt_signed_2dec(acc[0])},{fmt_signed_2dec(acc[1])})"
        return vel_str, acc_str

    def _nav_state_from_agent_input(self, agent_input: AgentInput):
        ego = agent_input.ego_statuses[-1]
        navigation_info = map_command_to_direction(np.array(ego.driving_command))
        vel = ego.ego_velocity.tolist()
        acc = ego.ego_acceleration.tolist()
        vel_str = f"({fmt_signed_2dec(vel[0])},{fmt_signed_2dec(vel[1])})"
        acc_str = f"({fmt_signed_2dec(acc[0])},{fmt_signed_2dec(acc[1])})"
        return navigation_info, (vel_str, acc_str)

    def _build_solver_inputs(self, agent_input: AgentInput, scene=None):
        """Prepare all inputs needed for the DFM solver.

        Returns (x_init, data_info) ready to pass to self.solver.sample().
        """
        if scene is not None:
            navigation_info = self.get_current_navigation_infomation(scene)
            state = self.get_ego_status(scene)
        else:
            navigation_info, state = self._nav_state_from_agent_input(agent_input)

        raw_img = agent_input.cameras[-1].cam_f0.image
        if isinstance(raw_img, np.ndarray):
            img = Image.fromarray(raw_img.astype(np.uint8)).convert("RGB")
        else:
            img = Image.open(str(raw_img)).convert("RGB")

        lidar_frame = agent_input.lidars[-1]
        has_bev = lidar_frame is not None and lidar_frame.lidar_pc is not None
        bev_points = lidar_frame.lidar_pc.T if has_bev else None  # [N, 6]

        init_pos_str = "(0.00,0.00)"
        conversation = [
            {
                "role": "User",
                "content": (
                    "Here is front views of a driving vehicle:\n"
                    "<image_placeholder>\n"
                    "<bev_placeholder>\n"
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

        bev_id = self.vl_chat_processor.bev_id
        input_ids = self.vl_chat_processor.add_bev_token(
            bev_indices=(input_ids == bev_id).nonzero(),
            input_ids=input_ids,
        )

        total_special = IMG_LEN + BEV_LEN
        rows_to_pad = max(self.txt_max_length + total_special - input_ids.shape[0], 100)
        input_ids = torch.cat(
            [input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0
        )
        attention_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)

        image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
        input_ids[torch.where(image_expanded_token_mask == 1)[0]] = 0

        bev_expanded_token_mask = (input_ids == bev_id).to(dtype=int)
        input_ids[torch.where(bev_expanded_token_mask == 1)[0]] = 0

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
        data_info['bev_points'] = [bev_points] * self.batch_size

        input_ids = input_ids.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        x_0_txt = torch.randint(self.vocabulary_size_txt, input_ids.shape, dtype=torch.long, device=self.device)
        x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])

        return x_init, data_info

    def _tokens_to_trajectory(self, traj_tokens: torch.Tensor) -> Trajectory:
        """Decode a 1-D trajectory token sequence → Trajectory (with heading)."""
        sentences = self.vl_chat_processor.tokenizer.decode(
            traj_tokens.tolist(), skip_special_tokens=True
        )
        nums = extract_num_list(sentences.strip())
        traj_xy_np = np.array(nums, dtype=np.float32).reshape(8, 2)

        # Fix corrupted waypoints by extrapolation
        for i in range(1, 8):
            prev_x = traj_xy_np[i - 1, 0]
            curr_x = traj_xy_np[i, 0]
            drop_threshold = max(3.0, prev_x * 0.4)
            if prev_x - curr_x > drop_threshold and prev_x > 2.0:
                if i >= 2:
                    dx = traj_xy_np[i - 1, 0] - traj_xy_np[i - 2, 0]
                    dy = traj_xy_np[i - 1, 1] - traj_xy_np[i - 2, 1]
                else:
                    dx = traj_xy_np[i - 1, 0] / i
                    dy = traj_xy_np[i - 1, 1] / i
                for j in range(i, 8):
                    traj_xy_np[j, 0] = traj_xy_np[j - 1, 0] + dx
                    traj_xy_np[j, 1] = traj_xy_np[j - 1, 1] + dy
                break

        traj_xy_t = torch.from_numpy(traj_xy_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            heading_pred = self.heading_model(traj_xy_t).cpu().squeeze().numpy()

        poses = np.concatenate([traj_xy_np, heading_pred.reshape(-1, 1)], axis=1).astype(np.float32)
        return Trajectory(poses)

    # ── Section 3.3: Perception-Oriented Trajectory Refinement ───────────────

    def _ego_to_bev_px(self, x: float, y: float):
        """Ego-frame metres (x=fwd, y=left) → BEV pixel (col, row) floats."""
        col = (x + BEV_RANGE) / BEV_RES
        row = (BEV_RANGE - y) / BEV_RES
        return col, row

    def _get_bev_perception(self, data_info, scene=None, max_voxels: int = 25000):
        """Run BEV encoder → seg_probs [6,180,180] and det_probs [180,180] numpy arrays.

        Returns continuous heatmaps (not discrete peaks) so that _residuals_torch
        can use grid_sample — consistent with how Gamma_risk is computed in training.
        Tries offline BEV cache first (keyed by scene token) to avoid spconv crashes.
        Falls back to live BEV encoder if no cache hit.
        """
        # ── Try precomputed cache ────────────────────────────────────────────
        if self.bev_cache_dir and scene is not None:
            token = scene.scene_metadata.initial_token
            cache_path = os.path.join(self.bev_cache_dir, f"{token}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                seg_probs = data['seg_probs'].astype(np.float32)   # [6,180,180] softmax
                det_probs = data['det_probs'][0].astype(np.float32) # [180,180] sigmoid
                return seg_probs, det_probs

        # ── Fall back to live BEV encoder ────────────────────────────────────
        bev_points = data_info['bev_points'][0]
        if bev_points is None:
            return None, None
        try:
            with torch.no_grad():
                bev_out = self.model.bev_encoder([bev_points], max_voxels=max_voxels)
        except Exception as e:
            print(f'[Section3.3] BEV perception failed: {e}')
            return None, None

        if bev_out is None:
            return None, None
        seg_logits = bev_out['seg_logits']                                      # [1, 6, 180, 180]
        seg_probs  = torch.softmax(seg_logits[0].float(), dim=0).cpu().numpy()  # [6, 180, 180]
        det_probs  = torch.sigmoid(
            bev_out['det_output']['heatmap'][0, 0].float()
        ).cpu().numpy()                                                          # [180, 180]
        return seg_probs, det_probs

    def _extract_det_centers(self, heatmap, top_n=10, threshold=0.3, min_dist=10):
        """Extract peak detections from heatmap → ego-frame (x, y) metres."""
        centers = []
        h = heatmap.copy()
        for _ in range(top_n):
            idx = np.argmax(h)
            r, c = np.unravel_index(idx, h.shape)
            if h[r, c] < threshold:
                break
            x = float(c * BEV_RES - BEV_RANGE)
            y = float(BEV_RANGE - r * BEV_RES)
            centers.append((x, y))
            r0 = max(0, r - min_dist);        r1 = min(h.shape[0], r + min_dist + 1)
            c0 = max(0, c - min_dist);        c1 = min(h.shape[1], c + min_dist + 1)
            h[r0:r1, c0:c1] = 0.0
        return centers

    def _residuals_torch(self, tau, tau_prior_t, seg_probs_t, det_probs_t):
        """GPU residual vector r(τ) — paper Eqs.(6)(8)(9)(10).

        tau, tau_prior_t : [16] float32 on self.device
        seg_probs_t      : [6,180,180] float32 on self.device, or None
        det_probs_t      : [180,180] float32 on self.device, or None
                           Continuous sigmoid heatmap from BEV det head.
                           Uses grid_sample (same as training Gamma_risk) so that
                           inference and training formulations are consistent.
        Returns [60] float32 tensor on self.device.
        """
        pts   = tau.reshape(8, 2)
        prior = tau_prior_t.reshape(8, 2)
        x, y  = pts[:, 0], pts[:, 1]

        # shared BEV grid coords for grid_sample
        gu = ((x + BEV_RANGE) / BEV_RES / (BEV_SIZE - 1) * 2. - 1.).clamp(-1., 1.)
        gv = ((BEV_RANGE - y) / BEV_RES / (BEV_SIZE - 1) * 2. - 1.).clamp(-1., 1.)
        grid = torch.stack([gu, gv], dim=-1).view(1, 1, 8, 2)                 # [1,1,8,2]

        # Γ_prior (Eq.6): sqrt(W) · (τ_i - τ̃_i)  [8,2]
        r_prior = (W_PRIOR ** 0.5) * (pts - prior)

        # Γ_risk seg (Eq.8,9): sqrt(Σ_c P(c|xi,yi)·ω_c)  [8]
        r_seg = torch.zeros(8, device=self.device, dtype=torch.float32)
        if seg_probs_t is not None:
            sampled  = F.grid_sample(
                seg_probs_t.unsqueeze(0), grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            )                                                                   # [1,6,1,8]
            probs_at = sampled[0, :, 0, :].T                                   # [8,6]
            seg_w    = torch.tensor(_SEG_WEIGHTS_RISK, device=self.device, dtype=torch.float32)
            seg_cost = (probs_at * seg_w).sum(dim=1).clamp(min=0.)
            r_seg    = torch.sqrt(seg_cost)                                     # [8]

        # Γ_risk det (Eq.9 second term): grid_sample(det_probs) × σ_det  [8]
        # Consistent with training: continuous heatmap, not discrete NMS peaks.
        r_det = torch.zeros(8, device=self.device, dtype=torch.float32)
        if det_probs_t is not None:
            sampled_det = F.grid_sample(
                det_probs_t.unsqueeze(0).unsqueeze(0),                         # [1,1,180,180]
                grid, mode='bilinear', align_corners=True, padding_mode='border'
            )                                                                   # [1,1,1,8]
            det_cost = (sampled_det[0, 0, 0, :] * SIGMA_DET).clamp(min=0.)    # [8]
            r_det    = torch.sqrt(det_cost)                                     # [8]

        # Γ_smooth accel (Eq.10 term 2): τ_{i+1} - 2τ_i + τ_{i-1}  [7,2]
        p     = torch.cat([torch.zeros(1, 2, device=self.device, dtype=torch.float32), pts])
        accel = p[2:] - 2. * p[1:-1] + p[:-2]
        r_accel = (W_SMOOTH_A ** 0.5) * accel                                  # [7,2]

        # Γ_smooth curv (Eq.10 term 1): κ_i = t_{i+1} - t_i  [7,2]
        vel   = p[1:] - p[:-1]                                                 # [8,2]
        vel_n = torch.norm(vel, dim=1, keepdim=True) + 1e-6
        t_hat = vel / vel_n
        r_curv = (W_SMOOTH_C ** 0.5) * (t_hat[1:] - t_hat[:-1])              # [7,2]

        return torch.cat([r_prior.ravel(), r_seg, r_det,
                          r_accel.ravel(), r_curv.ravel()])                    # [60]

    def _refine_trajectory(self, traj_xy, seg_probs, det_probs):
        """GPU Levenberg-Marquardt following paper Eq.(11). max 50 iterations.

        Jacobian computed via central finite differences (16 params → 32 forward passes).
        Returns refined [8,2] float32 array, or original on failure.
        """
        tau_prior_t = torch.tensor(
            traj_xy.flatten(), dtype=torch.float32, device=self.device
        )
        seg_probs_t = (
            torch.tensor(seg_probs, dtype=torch.float32, device=self.device)
            if seg_probs is not None else None
        )
        det_probs_t = (
            torch.tensor(det_probs, dtype=torch.float32, device=self.device)
            if det_probs is not None else None
        )
        tau = tau_prior_t.clone()

        n   = 16                                      # number of optimisation variables
        lam = 1e-2                                    # initial LM damping
        eps = 1e-4                                    # finite-difference step
        I   = torch.eye(n, device=self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                for _ in range(50):
                    r    = self._residuals_torch(tau, tau_prior_t, seg_probs_t, det_probs_t)
                    cost = (r * r).sum()

                    # Jacobian J [60, 16] via central finite differences
                    J = torch.zeros(r.shape[0], n, device=self.device, dtype=torch.float32)
                    for i in range(n):
                        tp    = tau.clone(); tp[i] += eps
                        tm    = tau.clone(); tm[i] -= eps
                        J[:, i] = (
                            self._residuals_torch(tp, tau_prior_t, seg_probs_t, det_probs_t) -
                            self._residuals_torch(tm, tau_prior_t, seg_probs_t, det_probs_t)
                        ) / (2. * eps)

                    # LM step: solve (J^T J + λI) δ = -J^T r
                    A     = J.T @ J + lam * I
                    b     = -(J.T @ r)
                    delta = torch.linalg.solve(A, b)

                    tau_new  = tau + delta
                    r_new    = self._residuals_torch(tau_new, tau_prior_t, seg_probs_t, det_probs_t)
                    cost_new = (r_new * r_new).sum()

                    if cost_new < cost:
                        tau = tau_new
                        lam = max(lam * 0.1, 1e-7)
                    else:
                        lam = min(lam * 10., 1e7)
                        if lam > 1e6:
                            break
        except Exception:
            pass  # return best tau found so far

        return tau.cpu().numpy().reshape(8, 2).astype(np.float32)

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        """Section 3.3 Algorithm 1: DFM K-traj → BEV perception → LM refine×K → argmax score."""
        with torch.no_grad():
            x_init, data_info = self._build_solver_inputs(agent_input, scene)
            intermediates = self.solver.sample(
                x_init=x_init,
                step_size=1.0 / self.discrete_fm_steps,
                return_intermediates=True,
                div_free=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=0,
            )

        # Alg.1 line 7: T = {τ̃(1),...,τ̃(K)}
        K = self.discrete_fm_steps
        cand_xys = []
        for k in range(1, K + 1):
            traj = self._tokens_to_trajectory(intermediates[k])
            cand_xys.append(traj.poses[:, :2].copy())  # [8,2]

        # Alg.1 lines 10-11: BEV perception → seg [6,180,180] and det [180,180] cost maps
        seg_probs, det_probs = self._get_bev_perception(data_info, scene=scene)
        seg_probs_t = (
            torch.tensor(seg_probs, dtype=torch.float32, device=self.device)
            if seg_probs is not None else None
        )
        det_probs_t = (
            torch.tensor(det_probs, dtype=torch.float32, device=self.device)
            if det_probs is not None else None
        )

        # Alg.1 lines 12-15: LM optimise each τ̃(k), score_k = exp(-Γ(τ̃*(k))) (Eq.12)
        # Use τ̃(K) as the common reference prior for fair cross-candidate comparison:
        # early steps τ̃(1..K-1) decode to near-stationary noise and their own-prior Γ≈0,
        # causing argmin to incorrectly prefer them over the actual model output τ̃(K).
        tau_K_t = torch.tensor(cand_xys[-1].flatten(), dtype=torch.float32, device=self.device)
        best_gamma = float('inf')
        best_xy    = cand_xys[-1]   # fallback: τ̃(K) (cleanest DFM output)

        for xy in cand_xys:
            refined = self._refine_trajectory(xy, seg_probs, det_probs)
            with torch.no_grad():
                tau_ref_t = torch.tensor(refined.flatten(), dtype=torch.float32, device=self.device)
                # τ̃(K) as common prior → Γ_prior penalises candidates far from final output
                r     = self._residuals_torch(tau_ref_t, tau_K_t, seg_probs_t, det_probs_t)
                gamma = float((r * r).sum())
            if gamma < best_gamma:
                best_gamma = gamma
                best_xy    = refined

        # Alg.1 line 16: τ̃*_best → add heading
        traj_xy_t = torch.from_numpy(best_xy).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            heading_pred = self.heading_model(traj_xy_t).cpu().squeeze().numpy()
        poses = np.concatenate([best_xy, heading_pred.reshape(-1, 1)], axis=1).astype(np.float32)
        return Trajectory(poses)

    def compute_trajectory_multi(self, agent_input: AgentInput, scene=None) -> List[Trajectory]:
        """Return K candidate trajectories from K DFM denoising steps.

        Implements Algorithm 1 of the paper (Section 3.2):
          x_0 → x_1 → ... → x_K  (sequential CTMC chain)
          τ̃(k) = decode(x_k)  for k = 1..K

        Returns:
            List of K Trajectory objects, ordered coarse (k=1) to fine (k=K).
        """
        with torch.no_grad():
            x_init, data_info = self._build_solver_inputs(agent_input, scene)
            # intermediates: [K+1, traj_len]
            #   row 0   = x_init  (random noise, skip)
            #   row 1   = τ̃(1)   (after 1st CTMC step)
            #   ...
            #   row K   = τ̃(K)   (final step = same as compute_trajectory())
            intermediates = self.solver.sample(
                x_init=x_init,
                step_size=1.0 / self.discrete_fm_steps,
                return_intermediates=True,
                div_free=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=0,
            )

        # Skip row 0 (random x_init noise), decode rows 1..K
        candidates = []
        for k in range(1, self.discrete_fm_steps + 1):
            candidates.append(self._tokens_to_trajectory(intermediates[k]))

        return candidates  # length == discrete_fm_steps

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        return self.compute_trajectory(agent_input, scene=None)
