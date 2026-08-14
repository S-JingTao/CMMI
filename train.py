# LINT_ME
import argparse
import contextlib
import os
import logging
import shutil
import copy
import random

import numpy as np
import transformers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import diffusers
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fudoki.model import instantiate_model
from fudoki.janus.models import VLChatProcessor
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.data.navsim import SupervisedDataset, navsim_collate_fn
from flow_matching.utils.flow import get_source_distribution


logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def init_numeric_and_special_tokens(
    model,
    tokenizer,
    numeric_tokens,
    noise_scale: float = 0.01,
):
    emb = model.get_input_embeddings().weight
    device = emb.device
    dim = emb.shape[1]

    def tok_ids_for_text(text: str):
        # Get subword ids (no specials), filter out -100/None if any
        ids = tokenizer.encode(text,add_special_tokens=False)
        return [i for i in ids if isinstance(i, int) and i >= 0]

    # ---- Build a numeric "base" vector from digits and dot ----
    digit_ids = []
    for d in "0123456789":
        _ids = tok_ids_for_text(d)
        digit_ids.extend(_ids)
    dot_ids = tok_ids_for_text(".")

    base_chunks = []
    if digit_ids:
        base_chunks.append(emb[torch.tensor(digit_ids, device=device)].mean(dim=0))
    if dot_ids:
        base_chunks.append(emb[torch.tensor(dot_ids, device=device)].mean(dim=0))
    if base_chunks:
        numeric_base = torch.stack(base_chunks, dim=0).mean(dim=0)
    else:
        # fallback if tokenizer lacks digits/dot as standalone pieces
        numeric_base = torch.zeros(dim, device=device)

    # ---- Initialize numeric tokens ----
    for t in numeric_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid < 0:
            continue
        noise = noise_scale * torch.randn(dim, device=device)
        emb[tid] = numeric_base + noise


def training_step(
    model,
    x_1,
    source_distribution,
    data_info,
    path,
    time_epsilon = 0.001,
    loss_fn = CrossEntropyLoss(),
    stage="s1",
    vl_chat_processor=None,
    args=None,
):
    x_0 = source_distribution.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)

    if stage == "s1":
        x_t = x_1
    elif stage == "s2":
        # update emb layer when using num tokenizer
        path_sample = path.sample(x_0, x_1, t)
        x_t = path_sample.x_t
    else:
        x_t = None

    # text_token_mask==1 ==> generated text token
    x_t = x_t * data_info['text_token_mask'] + x_1 * (1 - data_info['text_token_mask'])
    data_info['understanding_img'] = data_info['understanding_img'].to(dtype=next(model.parameters()).dtype)

    _, txt_logits = model(x_t, data_info)

    b, _, c = txt_logits.shape
    mask = data_info['text_token_mask'].unsqueeze(-1).bool()
    txt_logits = txt_logits.masked_select(mask)
    txt_logits = txt_logits.view(b, -1, c)
    x_1 = x_1.masked_select(mask.squeeze(-1)).view(b, -1)

    loss = ce_loss = loss_fn(txt_logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
    loss_dict = {"ce_loss": ce_loss.detach().item()}

    if stage == "s2":
        start_mask = x_1 >= vl_chat_processor.num_start_id
        end_mask = x_1 <= vl_chat_processor.num_end_id - 1
        action_mask = start_mask & end_mask

        if action_mask.any():
            if args.l2_loss_weight > 0:
                pred_probabilities = F.softmax(txt_logits, dim=-1)
                pred_ids = torch.argmax(pred_probabilities, dim=-1)
                pred_num_ids = pred_ids.masked_select(action_mask)
                pred_nums = vl_chat_processor.min_num + (pred_num_ids - vl_chat_processor.num_start_id) * vl_chat_processor.interval
                pred_nums = torch.clip(pred_nums, vl_chat_processor.min_num, vl_chat_processor.max_num)

                tgt_num_ids = x_1.masked_select(action_mask) # [N]
                tgt_nums = vl_chat_processor.min_num + (tgt_num_ids - vl_chat_processor.num_start_id) * vl_chat_processor.interval

                l2_loss = args.l2_loss_weight + F.mse_loss(pred_nums, tgt_nums, reduction="mean")
                loss = loss + l2_loss
                loss_dict["l2_loss"] = l2_loss.detach().item()

    loss_dict["loss"] = loss.detach().item()
    return loss, loss_dict


def training_step_plan(
    model,
    x_1,
    source_distribution,
    data_info,
    path,
    vl_chat_processor,
    K: int = 3,
    diversity_margin: float = 2.0,
    diversity_weight: float = 0.1,
    monotonic_weight: float = 0.1,
    time_epsilon: float = 0.001,
    loss_fn=CrossEntropyLoss(),
    seg_weight: float = 1.0,
    det_weight: float = 1.0,
    gamma_risk_weight: float = 0.0,
    gamma_smooth_weight: float = 0.0,
):
    """
    Implements L_plan from paper Section 3.2 (eq. 4):

      L_plan = min_k CE(k)
             - div_weight  * Σ_{i<j} ||τ̃(i) - τ̃(j)||_2
             + mono_weight * Σ_{k=1}^{K-1} max(0, d_{k+1} - d_k)

    K trajectories are produced by corrupting x_1 with K sorted t-values
    (differentiable proxy for K sequential CTMC steps at inference).
    Soft trajectories are decoded as expected numeric values (fully differentiable).

    The diversity term is - Σ_{i≠j} ||τ̃(i)-τ̃(j)||_2 (negative L2 sum):
    minimising the loss maximises pairwise distance, preventing trajectory collapse.
    `diversity_margin` is not used in this formulation but kept as a parameter
    for potential future use.
    """
    B      = x_1.shape[0]
    device = x_1.device

    # Numeric vocab setup
    num_start  = vl_chat_processor.num_start_id
    num_end    = vl_chat_processor.num_end_id
    num_vocab  = num_end - num_start + 1   # inclusive: num_start..num_end
    num_values = (vl_chat_processor.min_num
                  + torch.arange(num_vocab, dtype=torch.float32, device=device)
                  * vl_chat_processor.interval)  # [num_vocab]

    text_mask = data_info['text_token_mask']          # [B, seq_len]
    mask_3d   = text_mask.unsqueeze(-1).bool()
    data_info['understanding_img'] = data_info['understanding_img'].to(
        dtype=next(model.parameters()).dtype
    )

    # Ground-truth trajectory token ids [B, traj_len]
    x_1_traj = x_1.masked_select(text_mask.bool()).view(B, -1)

    # Numeric token positions (use first sample; format is the same for all)
    is_num        = (x_1_traj[0] >= num_start) & (x_1_traj[0] <= num_end)
    num_positions = is_num.nonzero(as_tuple=True)[0]  # expected [16]
    n_waypoint_coords = 16   # 8 waypoints × 2 coords
    if len(num_positions) < n_waypoint_coords:
        # Fewer numeric tokens than expected — fall back to plain CE loss
        x_t = x_1 * text_mask + x_1 * (1 - text_mask)
        _, txt_logits = model(x_t, data_info)
        traj_logits = txt_logits.masked_select(mask_3d).view(B, -1, txt_logits.shape[-1])
        ce_loss = loss_fn(traj_logits.flatten(0, 1), x_1_traj.flatten(0, 1)).mean()
        return ce_loss, {"loss": ce_loss.item(), "ce_loss": ce_loss.item(),
                         "div_loss": 0.0, "mono_loss": 0.0}
    num_positions = num_positions[:n_waypoint_coords]   # take first 16 if more

    # GT in float space [B, 8, 2]
    gt_num_ids = x_1_traj[:, num_positions].float()
    gt_nums    = (gt_num_ids - num_start) * vl_chat_processor.interval + vl_chat_processor.min_num
    gt_traj    = gt_nums.reshape(B, 8, 2)

    # ── BEV features: precomputed cache ONLY (live encoder removed — causes cudaErrorIllegalAddress) ──
    # NavSim LiDAR has 47k-65k voxels/frame; spconv native_pair crashes the CUDA context permanently.
    # Samples without cache simply skip Γ_risk (2.9% of training data).
    _bev_seg_logits = None
    _bev_det_logits = None
    _seg_probs      = None
    _det_probs      = None
    _need_bev = (gamma_risk_weight > 0) or (seg_weight > 0) or (det_weight > 0)
    if _need_bev:
        _cached_seg = data_info.get('bev_seg_probs', None)
        _cached_det = data_info.get('bev_det_probs', None)
        if _cached_seg is not None:
            _seg_probs = _cached_seg.float().to(device)   # [B,6,180,180]
            _det_probs = _cached_det.float().to(device) if _cached_det is not None else None

    # K ascending t values (low t = noisier = earlier step; high t = cleaner = later step)
    ts = torch.sort(torch.rand(K, device=device) * (1.0 - time_epsilon))[0]

    x_0 = source_distribution.sample_like(x_1)

    soft_trajs = []
    ce_losses  = []

    for k in range(K):
        t_k         = ts[k].expand(B)
        path_sample = path.sample(x_0, x_1, t_k)
        x_t         = path_sample.x_t
        x_t         = x_t * text_mask + x_1 * (1 - text_mask)

        _, txt_logits = model(x_t, data_info)          # [B, seq_len, vocab]

        # Extract only the trajectory token positions [B, traj_len, vocab]
        traj_logits = txt_logits.masked_select(mask_3d).view(B, -1, txt_logits.shape[-1])
        del txt_logits                                  # free full seq logits immediately

        # ── CE loss for this k (computed now, graph kept, tensor freed) ──
        ce_k = loss_fn(traj_logits.flatten(0, 1), x_1_traj.flatten(0, 1))
        ce_losses.append(ce_k)

        # ── Soft decode: expected value over numeric tokens ──────────────
        num_logits = traj_logits[:, num_positions, num_start:num_end+1]  # [B, 16, num_vocab]
        del traj_logits                                 # free traj logits after soft decode
        num_probs  = F.softmax(num_logits, dim=-1)                      # [B, 16, num_vocab]
        soft_vals  = (num_probs * num_values).sum(dim=-1)               # [B, 16]
        soft_trajs.append(soft_vals.reshape(B, 8, 2))                   # [B, 8, 2]

    # ── Term 1: min_k CE ────────────────────────────────────────────────
    loss_mink = torch.stack(ce_losses).min()            # [K] → scalar

    # ── Term 2: Diversity  - Σ_{i≠j} ||τ̃(i) - τ̃(j)||_2 ──────────────────
    # Negative: minimising loss = maximising pairwise distance = diverse trajectories.
    div_loss = torch.zeros(1, device=device, requires_grad=True)
    for i in range(K):
        for j in range(i + 1, K):
            dist     = (soft_trajs[i] - soft_trajs[j]).norm(dim=-1).mean()
            div_loss = div_loss - dist          # subtract to maximise diversity
    n_pairs  = max(K * (K - 1) / 2, 1)
    div_loss = div_loss / n_pairs
    div_loss = div_loss.clamp(min=-1.0)   # cap diversity contribution to CE scale

    # ── Term 3: Monotonic refinement ─────────────────────────────────────
    # Σ_k max(0, d_{k+1} - d_k)  — later steps must not be farther from GT
    dists = [
        (soft_trajs[k] - gt_traj).norm(dim=-1).mean(dim=-1)  # [B]
        for k in range(K)
    ]
    mono_loss = torch.zeros(1, device=device, requires_grad=True)
    for k in range(K - 1):
        mono_loss = mono_loss + F.relu(dists[k + 1] - dists[k]).mean()
    mono_loss = mono_loss / max(K - 1, 1)

    loss = loss_mink + diversity_weight * div_loss + monotonic_weight * mono_loss

    # ── Γ_risk: Σ_c P(c|x,y)·ω_c  +  σ_det·p_det(x,y)   (paper Eq. 9) ─────────
    # seg cost = weighted BEV semantic class probs at each trajectory point
    # det cost = detection heatmap probability × σ_det (Gaussian distance field approx.)
    gamma_risk_loss_val = 0.0
    if gamma_risk_weight > 0 and (_seg_probs is not None or _det_probs is not None):
        _SEG_W     = torch.tensor([0.0, 0.5, 1.0, 0.3, 2.0, 0.0],
                                   dtype=torch.float32, device=device)
        _DET_W     = 2.0   # σ_det: same scale as vehicle class weight in seg
        _BEV_RANGE = 54.0; _BEV_RES = 0.6; _BEV_SIZE = 180
        _gamma_risks = []
        for k in range(K):
            traj_k = soft_trajs[k]                     # [B, 8, 2]
            _x = traj_k[:, :, 0]                       # forward  [B, 8]
            _y = traj_k[:, :, 1]                       # lateral  [B, 8]
            gu = ((_x + _BEV_RANGE) / _BEV_RES / (_BEV_SIZE - 1) * 2.0 - 1.0).clamp(-1., 1.)
            gv = ((_BEV_RANGE - _y) / _BEV_RES / (_BEV_SIZE - 1) * 2.0 - 1.0).clamp(-1., 1.)
            grid = torch.stack([gu, gv], dim=-1).unsqueeze(1)   # [B, 1, 8, 2]

            # Semantic cost: Σ_c P(c|x,y)·ω_c
            if _seg_probs is not None:
                sampled_seg = F.grid_sample(_seg_probs, grid, mode='bilinear', align_corners=True)
                probs_at    = sampled_seg[:, :, 0, :].permute(0, 2, 1)    # [B, 8, 6]
                seg_cost    = (probs_at * _SEG_W).sum(dim=-1).clamp(min=0.)  # [B, 8]
            else:
                seg_cost = torch.zeros(B, 8, device=device)

            # Detection cost: σ_det · p_det(x,y)
            if _det_probs is not None:
                sampled_det = F.grid_sample(_det_probs, grid, mode='bilinear', align_corners=True)
                det_cost    = sampled_det[:, 0, 0, :] * _DET_W            # [B, 8]
            else:
                det_cost = torch.zeros(B, 8, device=device)

            r_total = torch.sqrt(seg_cost + det_cost + 1e-6)              # [B, 8]
            _gamma_risks.append(r_total.mean())
        gamma_risk_loss = torch.stack(_gamma_risks).mean()
        loss = loss + gamma_risk_weight * gamma_risk_loss
        gamma_risk_loss_val = gamma_risk_loss.detach().item()

    # ── Γ_smooth: penalise acceleration across soft trajectory candidates ────────
    # Γ_smooth = mean_k mean_{b,t} ||accel_t||  (dt=0.5 s between waypoints)
    gamma_smooth_loss_val = 0.0
    if gamma_smooth_weight > 0:
        _dt = 0.5
        _gamma_smooths = []
        for k in range(K):
            traj_k = soft_trajs[k]                                     # [B, 8, 2]
            vel    = (traj_k[:, 1:, :] - traj_k[:, :-1, :]) / _dt    # [B, 7, 2]
            accel  = (vel[:, 1:, :]    - vel[:, :-1, :])    / _dt    # [B, 6, 2]
            _gamma_smooths.append(accel.norm(dim=-1).mean())
        gamma_smooth_loss = torch.stack(_gamma_smooths).mean()
        loss = loss + gamma_smooth_weight * gamma_smooth_loss
        gamma_smooth_loss_val = gamma_smooth_loss.detach().item()

    # ── Section 3.3: BEV seg + det supervision (reuse pre-computed BEV output) ───
    seg_loss_val = 0.0
    det_loss_val = 0.0
    if (seg_weight > 0 or det_weight > 0) and _bev_seg_logits is not None and _bev_det_logits is not None:
        gt_seg = data_info.get('gt_seg', None)   # [B,180,180] int64 or None
        gt_det = data_info.get('gt_det', None)   # [B,180,180] float32 or None
        if gt_seg is not None and gt_det is not None:
            try:
                gt_seg_dev = gt_seg.to(device)
                gt_det_dev = gt_det.to(device)

                seg_loss_t = F.cross_entropy(_bev_seg_logits.float(), gt_seg_dev)

                det_prob = torch.sigmoid(_bev_det_logits[:, 0].float())
                pos_mask = (gt_det_dev > 0.5).float()
                neg_mask = 1.0 - pos_mask
                pos_loss = -pos_mask * (1 - det_prob) ** 2 * torch.log(det_prob + 1e-6)
                neg_loss = -neg_mask * (1 - gt_det_dev) ** 4 * det_prob ** 2 \
                           * torch.log(1 - det_prob + 1e-6)
                num_pos_actual = pos_mask.sum()
                if num_pos_actual > 0:
                    det_loss_t = (pos_loss.sum() + neg_loss.sum()) / num_pos_actual
                    loss = loss + seg_weight * seg_loss_t + det_weight * det_loss_t
                    det_loss_val = det_loss_t.detach().item()
                else:
                    loss = loss + seg_weight * seg_loss_t
                    det_loss_val = 0.0
                seg_loss_val = seg_loss_t.detach().item()
            except Exception:
                pass  # spconv may fail on some samples; skip silently

    loss_dict = {
        "loss":              loss.detach().item(),
        "ce_loss":           loss_mink.detach().item(),
        "div_loss":          div_loss.detach().item(),
        "mono_loss":         mono_loss.detach().item(),
        "gamma_risk_loss":   gamma_risk_loss_val,
        "gamma_smooth_loss": gamma_smooth_loss_val,
        "seg_loss":          seg_loss_val,
        "det_loss":          det_loss_val,
    }
    return loss, loss_dict


def training_step_seg_det(
    model,
    data_info: dict,
    device,
    seg_weight: float = 1.0,
    det_weight: float = 1.0,
):
    """Phase 1: supervise only seg_head + det_head with GT labels. No LLM forward."""
    bev_points = data_info.get('bev_points', None)
    gt_seg     = data_info.get('gt_seg',    None)
    gt_det     = data_info.get('gt_det',    None)

    # DeepSpeed requires loss.grad_fn is not None — leaf tensors fail.
    # Always produce dummy via seg_head params so grad_fn exists.
    def _param_zero():
        return sum(0.0 * p.sum() for p in model.bev_encoder.seg_head.parameters())

    has_valid_points = (
        bev_points is not None
        and isinstance(bev_points, (list, tuple))
        and any(p is not None for p in bev_points)
    )
    if not has_valid_points:
        return _param_zero(), {"loss": 0.0, "seg_loss": 0.0, "det_loss": 0.0}

    try:
        bev_out    = model.bev_encoder(bev_points)
        seg_logits = bev_out['seg_logits']                  # [B, 6, 180, 180]
        det_logits = bev_out['det_output']['heatmap']       # [B, 1, 180, 180]

        if gt_seg is None or gt_det is None:
            # No GT for this batch — zero loss still connected to graph
            dummy = 0.0 * (seg_logits.sum() + det_logits.sum())
            return dummy, {"loss": 0.0, "seg_loss": 0.0, "det_loss": 0.0}

        gt_seg_dev = gt_seg.to(device)                      # [B, 180, 180] int64
        gt_det_dev = gt_det.to(device)                      # [B, 180, 180] float32

        seg_loss = F.cross_entropy(seg_logits.float(), gt_seg_dev)

        det_prob = torch.sigmoid(det_logits[:, 0].float())
        pos_mask = (gt_det_dev > 0.5).float()
        neg_mask = 1.0 - pos_mask
        pos_loss = -pos_mask * (1 - det_prob) ** 2 * torch.log(det_prob + 1e-6)
        neg_loss = -neg_mask * (1 - gt_det_dev) ** 4 * det_prob ** 2 \
                   * torch.log(1 - det_prob + 1e-6)
        num_pos_actual = pos_mask.sum()
        if num_pos_actual > 0:
            det_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos_actual
        else:
            # No vehicle centers in this sample — skip det loss to prevent
            # the unbalanced neg_loss / 1 from driving all logits to -inf.
            det_loss = 0.0 * det_logits.sum()  # zero but graph-connected

        loss = seg_weight * seg_loss + det_weight * det_loss
        return loss, {
            "loss":     loss.detach().item(),
            "seg_loss": seg_loss.detach().item(),
            "det_loss": det_loss.detach().item(),
        }
    except Exception as e:
        logger.warning(f"seg_det training step exception: {e}")
        return _param_zero(), {"loss": 0.0, "seg_loss": 0.0, "det_loss": 0.0}


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # random seed
    seed = args.seed + accelerator.process_index
    if args.random_seed:
        seed = seed + random.randint(0, 500)
    set_seed(seed)
    logger.info(f"accelerator.process_index: {accelerator.process_index},   seed: {seed} \n")

    # work dir — all ranks create the dir (exist_ok=True) to avoid needing a barrier
    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        config_path = os.path.join(args.output_dir, "config.yaml")
        OmegaConf.save(args, config_path)
        accelerate_config_path = os.path.join(args.output_dir, "accelerate_config_ds2.yaml")
        shutil.copyfile(
            "./config/accelerate_config_ds2.yaml",
            accelerate_config_path
        )
    # skip wait_for_everyone() here to avoid NCCL/NVML driver mismatch at startup

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # prepare dataset
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    if args.use_quantize:
        origin_len = len(vl_chat_processor.tokenizer)
        num_tokens = [f"{x:.2f}" for x in np.linspace(-100, 100, 20001)]
        num_tokens_length = len(num_tokens)
        vl_chat_processor.tokenizer.add_tokens(num_tokens)

        vl_chat_processor.num_start_id = origin_len
        vl_chat_processor.num_end_id = origin_len + num_tokens_length - 1
        vl_chat_processor.min_num = -100
        vl_chat_processor.max_num = 100
        vl_chat_processor.interval = 0.01

        logger.info(f"Total number tokens: {num_tokens_length}")

    # data
    dataset = SupervisedDataset(
        data_list=args.data_list,
        vl_chat_processor=vl_chat_processor,
        txt_max_length=args.txt_max_length,
        bev_cache_dir=getattr(args, 'bev_cache_dir', ''),
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=navsim_collate_fn,
    )
    logger.info(f"Max txt length: {args.txt_max_length}")
    logger.info(f"Total data samples: {len(dataset)}")

    # prepare model
    stage = args.stage
    logger.info(f"Training stege: {stage}")
    model = instantiate_model(
        args.pretrain_model_path
    ).to(weight_dtype)
    model.uncond_prob = args.uncond_prob

    if os.path.exists(args.pretrain_path):
        sd = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(sd, strict=True)
        model = model.to(weight_dtype)
        logger.info(f"Loading pretrain ckpt from {args.pretrain_path}")

    bev_ckpt_path = getattr(args, 'bev_ckpt_path', '')
    if bev_ckpt_path and os.path.exists(bev_ckpt_path):
        model.add_bev_encoder(bev_ckpt_path, freeze_backbone=True)
        logger.info(f"BEV encoder loaded from {bev_ckpt_path}")
    # Optionally disable BEV token injection into the LLM (use BEV only for seg/det).
    if not getattr(args, 'use_bev_tokens', True):
        model.use_bev_tokens = False
        logger.info("BEV token injection disabled (use_bev_tokens=false); seg/det still active")

    if stage == "s1":
        model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)
        init_numeric_and_special_tokens(
            model.language_model,
            vl_chat_processor.tokenizer,
            numeric_tokens=num_tokens
        )
    elif stage == "s2":
        if os.path.exists(args.new_embedding_path):
            old_emb = copy.deepcopy(model.language_model.get_input_embeddings())

            with torch.serialization.safe_globals([torch.nn.modules.sparse.Embedding]):
                new_emb_state = torch.load(args.new_embedding_path, map_location="cpu")
                logger.info(f"Loading new embedding from {args.new_embedding_path}")

            if isinstance(new_emb_state, dict) and "weight" in new_emb_state:
                weight = new_emb_state["weight"]
                new_emb = torch.nn.Embedding(weight.size(0), weight.size(1))
                new_emb.load_state_dict(new_emb_state)
            else:
                new_emb = new_emb_state

            model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)

            # origin_len = old_emb.weight.shape[0]
            origin_len = vl_chat_processor.num_start_id
            new_emb.weight.data[:origin_len, :] = old_emb.weight.data[:origin_len, :]

            model.language_model.set_input_embeddings(new_emb)

        if os.path.exists(args.ckpt_path):
            if model.language_model.model.embed_tokens.weight.shape[0] != args.vocab_size + num_tokens_length:
                model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)
            if args.ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file as st_load_file
                sd = st_load_file(args.ckpt_path, device='cpu')
            else:
                sd = torch.load(args.ckpt_path, map_location='cpu')
            # Filter shape-mismatched keys when seg/det heads have new architecture.
            # Phase 1: seg_det_pretrain=True, heads re-initialized from scratch.
            # Phase 2: seg_det_ckpt_path set, old det_head in ckpt will be overwritten anyway.
            if getattr(args, 'seg_det_pretrain', False) or getattr(args, 'seg_det_ckpt_path', ''):
                model_sd = model.state_dict()
                sd = {k: v for k, v in sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(sd, strict=True)
            model = model.to(weight_dtype)
            logger.info(f"Loading ckpt from {args.ckpt_path}")

    # Phase 2: load pretrained seg/det weights and keep them frozen
    seg_det_ckpt = getattr(args, 'seg_det_ckpt_path', '')
    if seg_det_ckpt and os.path.exists(seg_det_ckpt) and model.bev_encoder is not None:
        from safetensors.torch import load_file as st_load_file
        sd = st_load_file(seg_det_ckpt, device='cpu')
        seg_det_sd = {
            k[len('bev_encoder.'):]: v for k, v in sd.items()
            if k.startswith('bev_encoder.seg_head') or k.startswith('bev_encoder.det_head')
        }
        model.bev_encoder.load_state_dict(seg_det_sd, strict=False)
        logger.info(f"Loaded seg/det weights ({len(seg_det_sd)} keys) from {seg_det_ckpt}")

    # prepare path
    path = MixtureDiscreteSoftmaxProbPath(
        mode='text',
        embedding_path=args.text_embedding_path
    )
    if args.use_quantize:
        path.set_embedding(model.language_model.get_input_embeddings())
    else:
        logger.info("No quantize!")

    logger.info(f"path.a = {path.a}")
    logger.info(f"path.c = {path.c}")

    # set trainable params
    model.requires_grad_(False)
    seg_det_pretrain = getattr(args, 'seg_det_pretrain', False)
    if seg_det_pretrain:
        # Phase 1: only seg_head + det_head trainable, everything else frozen
        if model.bev_encoder is not None:
            model.bev_encoder.seg_head.requires_grad_(True)
            model.bev_encoder.det_head.requires_grad_(True)
    else:
        if stage == "s1":
            model.language_model.requires_grad_(False)
            model.language_model.model.embed_tokens.requires_grad_(True)
            model.language_model.lm_head.requires_grad_(True)
        elif stage == "s2":
            model.language_model.requires_grad_(True)
            if args.train_llm_emb:
                model.language_model.model.embed_tokens.requires_grad_(True)
            else:
                model.language_model.model.embed_tokens.requires_grad_(False)

        # Phase 2: BEV projector trainable; seg/det frozen (no requires_grad here)
        if model.bev_encoder is not None:
            model.bev_encoder.bev_projector.requires_grad_(True)

    trainable_params = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )

    # log trainable params
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable params: {name}")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_trainable/1e9:.3} B")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes
    )

    # Enable gradient checkpointing on inner LLM to reduce activation memory for K>2
    if getattr(args, 'plan_loss', False) and getattr(args, 'plan_k', 2) > 2:
        if hasattr(model, 'language_model'):
            model.language_model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'language_model'):
            model.module.language_model.gradient_checkpointing_enable()

    # accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    source_distribution = get_source_distribution(
        source_distribution=args.source_distribution,
        vocab_size=args.vocab_size + num_tokens_length if args.use_quantize else args.vocab_size,
    )


    global_step = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # training loop
    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs}")
        logger.info(f"training sample length: {len(dataloader)}")

        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                x_1 = batch["input_ids"].to(dtype=torch.long)

                if seg_det_pretrain:
                    # Phase 1: seg/det only, no LLM forward
                    loss, logs = training_step_seg_det(
                        model=model,
                        data_info=batch,
                        device=accelerator.device,
                        seg_weight=getattr(args, 'plan_seg_weight', 1.0),
                        det_weight=getattr(args, 'plan_det_weight', 1.0),
                    )
                elif getattr(args, 'plan_loss', False) and stage == "s2":
                    # Phase 2: L_plan + Γ_risk + Γ_smooth (seg/det weights=0)
                    loss, logs = training_step_plan(
                        x_1=x_1,
                        model=model,
                        source_distribution=source_distribution,
                        data_info=batch,
                        path=path,
                        vl_chat_processor=vl_chat_processor,
                        K=getattr(args, 'plan_k', 3),
                        diversity_margin=getattr(args, 'plan_div_margin', 2.0),
                        diversity_weight=getattr(args, 'plan_div_weight', 0.1),
                        monotonic_weight=getattr(args, 'plan_mono_weight', 0.1),
                        seg_weight=0.0,
                        det_weight=0.0,
                        gamma_risk_weight=getattr(args, 'plan_gamma_risk_weight', 0.0),
                        gamma_smooth_weight=getattr(args, 'plan_gamma_smooth_weight', 0.0),
                    )
                else:
                    loss, logs = training_step(
                        x_1=x_1,
                        model=model,
                        source_distribution=source_distribution,
                        data_info=batch,
                        path=path,
                        stage=stage,
                        vl_chat_processor=vl_chat_processor,
                        args=args,
                    )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # defragment after each optimizer step

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs["lr"] = optimizer.param_groups[0]['lr']

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    # 所有 rank 先同步，再由 main process 清理旧 checkpoint 并保存
                    # 原来 wait_for_everyone() 在 is_main_process 块内，只有 rank 0 调用，
                    # 其余 rank 跑到下一步 ALLREDUCE，rank 0 卡在 barrier → NCCL 600s 超时
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                if os.path.exists(removing_checkpoint):
                                    shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()  # 所有 rank 同步后再保存
                    if accelerator.is_main_process:
                        unwrap_net = accelerator.unwrap_model(model)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unwrap_net.save_pretrained(save_path, max_shard_size="20GB")
                        logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone()  # 等待主进程保存完成再继续训练

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("training completed!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_obs_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.load(args.config)

    # merge args
    args_dict = vars(args).copy()
    args_dict.pop("config", None)
    config_keys = set(config.keys())
    cli_keys = set(args_dict.keys())

    # check conflict
    conflict_keys = cli_keys & config_keys
    if conflict_keys:
        print(f"Args conflict: {conflict_keys}")

    # merge
    merged_config = OmegaConf.merge(OmegaConf.create(args_dict), config)
    args = merged_config

    # training
    main(args)
