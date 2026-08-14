"""
bev_encoder.py

BEVFusion LiDAR-only encoder with three outputs:
  1. bev_tokens  [B, 196, 2048]  → fed into VLA (BEV Token)
  2. seg_logits  [B, N_cls, H, W]  → BEV semantic segmentation (future use)
  3. det_output  dict  → 3D detection results (future use)

Uses BEVFusion source directly from bevfusion/ subdirectory.
BEVFusion backbone weights are loaded and frozen.
BEVProjector and SegHead are randomly initialized and trainable.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# ── Compatibility shims (must come before any bevfusion imports) ──────────
# Use direct file import to avoid triggering fudoki package __init__.py
import importlib
import importlib.util as _ilu
_shim_path = os.path.join(os.path.dirname(__file__), 'mmcv_shim.py')
_shim_spec = _ilu.spec_from_file_location('mmcv_shim', _shim_path)
_shim_mod  = _ilu.module_from_spec(_shim_spec)
_shim_spec.loader.exec_module(_shim_mod)
_shim_mod.install()

# ── Patch bevfusion's spconv → use installed spconv-cu124 ─────────────────
BEVFUSION_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bevfusion')
BEVFUSION_ROOT = os.path.abspath(BEVFUSION_ROOT)
if BEVFUSION_ROOT not in sys.path:
    sys.path.insert(0, BEVFUSION_ROOT)

import spconv.pytorch as _spconv_pytorch
import types as _types

# ── Implement SparseBasicBlock and make_sparse_convmodule directly ────────
# (avoids importing bevfusion sparse_block.py which has mmdet BasicBlock dependency)

class SparseBasicBlock(_spconv_pytorch.SparseModule):
    """Sparse residual block matching BEVFusion's SparseBasicBlock interface."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 conv_cfg=None, norm_cfg=None):
        super().__init__()
        norm_cfg = norm_cfg or {'type': 'BN1d', 'eps': 1e-3, 'momentum': 0.01}
        eps = norm_cfg.get('eps', 1e-3)
        mom = norm_cfg.get('momentum', 0.01)

        self.conv1 = _spconv_pytorch.SubMConv3d(
            inplanes, planes, 3, stride=1, padding=1, bias=False,
            indice_key=f'subm_{inplanes}_{planes}')
        self.bn1 = nn.BatchNorm1d(planes, eps=eps, momentum=mom)
        self.conv2 = _spconv_pytorch.SubMConv3d(
            planes, planes, 3, stride=1, padding=1, bias=False,
            indice_key=f'subm_{inplanes}_{planes}')
        self.bn2 = nn.BatchNorm1d(planes, eps=eps, momentum=mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = out.replace_feature(self.relu(out.features + identity))
        return out


def make_sparse_convmodule(in_channels, out_channels, kernel_size, indice_key,
                            stride=1, padding=0, conv_type='SubMConv3d',
                            norm_cfg=None, order=('conv', 'norm', 'act')):
    """Build a sparse conv module: [SparseConv] + [BN] + [ReLU]."""
    norm_cfg = norm_cfg or {'type': 'BN1d', 'eps': 1e-3, 'momentum': 0.01}
    eps = norm_cfg.get('eps', 1e-3)
    mom = norm_cfg.get('momentum', 0.01)

    conv_cls = getattr(_spconv_pytorch, conv_type, _spconv_pytorch.SubMConv3d)
    conv = conv_cls(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, bias=False,
                    indice_key=indice_key)
    norm = nn.BatchNorm1d(out_channels, eps=eps, momentum=mom)
    act  = nn.ReLU(inplace=True)

    layers = []
    for o in order:
        if o == 'conv': layers.append(conv)
        elif o == 'norm': layers.append(norm)
        elif o == 'act':  layers.append(act)

    return _spconv_pytorch.SparseSequential(*layers)


# mmdet3d package tree with our implementations
_mmdet3d         = _types.ModuleType('mmdet3d')
_mmdet3d_ops_pkg = _types.ModuleType('mmdet3d.ops')
_mmdet3d_ops_pkg.spconv                = _spconv_pytorch
_mmdet3d_ops_pkg.SparseBasicBlock      = SparseBasicBlock
_mmdet3d_ops_pkg.make_sparse_convmodule = make_sparse_convmodule
_mmdet3d.ops = _mmdet3d_ops_pkg
for k, v in (('mmdet3d', _mmdet3d), ('mmdet3d.ops', _mmdet3d_ops_pkg),
             ('mmdet3d.ops.spconv', _spconv_pytorch)):
    sys.modules[k] = v

# mmdet shim (sparse_encoder.py doesn't need mmdet, but second.py might check)
_mmdet     = _types.ModuleType('mmdet')
_mmdet_mod = _types.ModuleType('mmdet.models')
_Reg = type('Reg', (), {'register_module': lambda s, *a, **k: (lambda c: c)})()
_mmdet_mod.BACKBONES = _Reg
_mmdet_mod.NECKS     = _Reg
_mmdet.models = _mmdet_mod
for k, v in (('mmdet', _mmdet), ('mmdet.models', _mmdet_mod)):
    sys.modules.setdefault(k, v)

# Import SparseEncoder, SECOND backbone, SECONDFPN neck directly
def _load_module(name, rel_path):
    path = os.path.join(BEVFUSION_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_se_mod   = _load_module('mmdet3d.models.backbones.sparse_encoder',
                         'mmdet3d/models/backbones/sparse_encoder.py')
_sec_mod  = _load_module('mmdet3d.models.backbones.second',
                         'mmdet3d/models/backbones/second.py')
_neck_mod = _load_module('mmdet3d.models.necks.second',
                         'mmdet3d/models/necks/second.py')

SparseEncoder = _se_mod.SparseEncoder
SECOND        = _sec_mod.SECOND
SECONDFPN     = _neck_mod.SECONDFPN


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud preprocessing (matching BEVFusion config)
# ─────────────────────────────────────────────────────────────────────────────

POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
VOXEL_SIZE       = [0.075, 0.075, 0.2]
# sparse_shape = [H, W, D] = [Y, X, Z] as used in BEVFusion source
SPARSE_SHAPE     = [1440, 1440, 41]   # [Y=1440, X=1440, Z=41]
MAX_POINTS_PER_VOXEL = 10
MAX_VOXELS       = 120000


def preprocess_points(points: np.ndarray) -> np.ndarray:
    """
    Filter and prepare raw PCD points.
    Input:  [N, 6]  (x, y, z, intensity, ring, lidar_info)
    Output: [M, 5]  (x, y, z, intensity_norm, time=0)
    """
    xyz       = points[:, :3].astype(np.float32)
    intensity = (points[:, 3:4].astype(np.float32)) / 255.0

    pcr = POINT_CLOUD_RANGE
    mask = (
        (xyz[:, 0] >= pcr[0]) & (xyz[:, 0] < pcr[3]) &
        (xyz[:, 1] >= pcr[1]) & (xyz[:, 1] < pcr[4]) &
        (xyz[:, 2] >= pcr[2]) & (xyz[:, 2] < pcr[5])
    )
    xyz       = xyz[mask]
    intensity = intensity[mask]
    time      = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, intensity, time], axis=1)


def voxelize(points: np.ndarray,
             max_voxels: int = MAX_VOXELS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard voxelization.
    Coordinates returned in (z, y, x) order matching BEVFusion coors format
    (batch_idx, z, y, x) expected by SparseConvTensor.
    Returns:
        voxel_features [V, 5]
        voxel_coords   [V, 3]  in (z, y, x) order
    """
    vs   = np.array(VOXEL_SIZE, dtype=np.float32)
    pcr  = np.array(POINT_CLOUD_RANGE[:3], dtype=np.float32)

    coords_f = (points[:, :3] - pcr) / vs   # (N, 3) float grid x,y,z
    coords_i = coords_f.astype(np.int32)

    gX, gY, gZ = SPARSE_SHAPE[1], SPARSE_SHAPE[0], SPARSE_SHAPE[2]  # X, Y, Z sizes
    coords_i[:, 0] = np.clip(coords_i[:, 0], 0, gX - 1)  # x
    coords_i[:, 1] = np.clip(coords_i[:, 1], 0, gY - 1)  # y
    # z boundary fix: level-3 encoder has padding_z=0, so z=0,1 vanish; keep z in [2, gZ-5]
    coords_i[:, 2] = np.clip(coords_i[:, 2], 2, gZ - 5)  # z in [2, 36]

    # Encode (x, y, z) → unique id
    voxel_ids = (coords_i[:, 0].astype(np.int64) +
                 coords_i[:, 1].astype(np.int64) * gX +
                 coords_i[:, 2].astype(np.int64) * gX * gY)

    unique_ids, inverse = np.unique(voxel_ids, return_inverse=True)
    if len(unique_ids) > max_voxels:
        unique_ids = unique_ids[:max_voxels]
        valid      = inverse < max_voxels
        points     = points[valid]
        inverse    = inverse[valid]

    V = len(unique_ids)
    vf = np.zeros((V, points.shape[1]), dtype=np.float32)
    np.add.at(vf, inverse[:len(points)], points[:len(points)])
    cnt = np.bincount(inverse[:len(points)], minlength=V).astype(np.float32)
    vf /= np.maximum(cnt[:, None], 1.0)

    # Decode ids → (x, y, z)
    vx = (unique_ids % gX).astype(np.int32)
    vy = ((unique_ids // gX) % gY).astype(np.int32)
    vz = (unique_ids // (gX * gY)).astype(np.int32)

    # BEVFusion expects coors = (batch, z, y, x)
    voxel_coords = np.stack([vz, vy, vx], axis=1)  # [V, 3]
    return vf, voxel_coords


# ─────────────────────────────────────────────────────────────────────────────
# Output heads
# ─────────────────────────────────────────────────────────────────────────────

SEG_CLASSES = ['drivable_area', 'ped_crossing', 'walkway',
               'lane_divider', 'vehicle', 'background']


class BEVProjector(nn.Module):
    """
    Output 1: BEV tokens for VLA.
    [B, C, H, W] → AdaptiveAvgPool(14×14) → [B, 196, C] → MLP → [B, 196, 2048]
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 2048, grid: int = 14):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.pool(x)                                     # [B, C, 14, 14]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)      # [B, 196, C]
        return self.proj(x)                                  # [B, 196, 2048]


class BEVSegHead(nn.Module):
    """
    Output 2: BEV semantic segmentation.
    [B, C, H, W] → [B, N_cls, H, W]
    """
    def __init__(self, in_channels: int = 256, n_classes: int = len(SEG_CLASSES)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # [B, N_cls, H, W]


class SimpleCenterHead(nn.Module):
    """
    Output 3: Single-channel obstacle heatmap head (GT-supervised).
    Outputs raw logits [B, 1, H, W]; apply sigmoid for probability.
    [B, C, H, W] → dict{'heatmap': [B, 1, H, W]}
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.shared_conv  = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.relu(self.shared_conv(x))
        return {'heatmap': self.heatmap_head(x)}  # [B, 1, H, W] logits


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BEVFusionEncoder(nn.Module):
    """
    BEVFusion LiDAR-only encoder (uses original bevfusion source code).

    Architecture (from BEVFusion config voxelnet_0p075.yaml):
      SparseEncoder     → [B, 256, 180, 180]   (pretrained, frozen)
      SECOND backbone   → multi-scale features   (pretrained, frozen)
      SECONDFPN neck    → [B, 256, 180, 180]    (pretrained, frozen)
      ├─ BEVProjector   → [B, 196, 2048]         (trainable)
      ├─ BEVSegHead     → [B, 6, 180, 180]       (trainable)
      └─ SimpleCenterHead → {'heatmap': ...}     (pretrained/trainable)

    Args:
        ckpt_path (str): Path to pretrained BEVFusion checkpoint.
        freeze_backbone (bool): Freeze SparseEncoder/backbone/neck. Default True.
        bev_channels (int): Channels of the BEV feature (neck output). Default 256.
    """

    def __init__(
        self,
        ckpt_path: str,
        freeze_backbone: bool = True,
        bev_channels: int = 256,
    ):
        super().__init__()

        # ── BEVFusion backbone components (original source) ──────────────
        self.sparse_encoder = SparseEncoder(
            in_channels=5,
            sparse_shape=SPARSE_SHAPE,
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
            block_type='basicblock',
        )
        # Force Native algorithm on all sparse convs to avoid MaskImplicitGemm crash
        # MaskImplicitGemm fails on certain input sizes; Native is slower but always correct
        from spconv.core import ConvAlgo as _ConvAlgo
        for m in self.sparse_encoder.modules():
            if hasattr(m, 'algo'):
                m.algo = _ConvAlgo.Native

        self.backbone = SECOND(
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
        )

        # Neck config from BEVFusion secfpn/default.yaml:
        #   in_channels=[128, 256]  (backbone blocks.0=128ch, blocks.1=256ch)
        #   out_channels=[256, 256]
        #   use_conv_for_no_stride=True  → deblocks.0 = Conv2d(128→256, k=1)
        #   deblocks.0 weight: (256,128,1,1) ✓  deblocks.1 weight: (256,256,2,2) ✓
        self.neck = SECONDFPN(
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            use_conv_for_no_stride=True,
        )

        # bev_channels = 256+256 = 512 (neck concat output)
        _neck_out = 512
        self._neck_out = _neck_out  # stored for fallback path

        # ── New trainable output heads ────────────────────────────────────
        self.bev_projector = BEVProjector(in_dim=_neck_out, out_dim=2048)
        self.seg_head      = BEVSegHead(in_channels=_neck_out)
        self.det_head      = SimpleCenterHead(in_channels=_neck_out)

        # ── Load pretrained weights ───────────────────────────────────────
        self._load_pretrained(ckpt_path)

        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd   = ckpt['state_dict']

        remap = {
            'pts_middle_encoder.': 'sparse_encoder.',
            'pts_backbone.':        'backbone.',
            'pts_neck.':            'neck.',
            'bbox_head.':           'det_head.',
        }
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for old, new in remap.items():
                if k.startswith(old):
                    nk = new + k[len(old):]
                    break
            new_sd[nk] = v

        miss, unexpected = self.load_state_dict(new_sd, strict=False)
        backbone_miss = [k for k in miss
                         if not any(k.startswith(h)
                                    for h in ('bev_projector', 'seg_head', 'det_head'))]
        print(f'[BEVFusionEncoder] loaded — backbone missing: {len(backbone_miss)}, '
              f'new heads missing: {len(miss) - len(backbone_miss)}')
        if backbone_miss:
            print(f'  backbone missing (first 3): {backbone_miss[:3]}')

    def _freeze_backbone(self):
        for m in (self.sparse_encoder, self.backbone, self.neck):
            for p in m.parameters():
                p.requires_grad = False
        print('[BEVFusionEncoder] backbone/neck frozen')

    @staticmethod
    def batch_voxelize(points_list: List[np.ndarray],
                       max_voxels: int = MAX_VOXELS,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_vf, all_vc = [], []
        for b_idx, pts in enumerate(points_list):
            pts = preprocess_points(pts)
            vf, vc = voxelize(pts, max_voxels=max_voxels)
            batch_col = np.full((len(vc), 1), b_idx, dtype=np.int32)
            all_vf.append(vf)
            all_vc.append(np.hstack([batch_col, vc]))   # (V, 4): batch,z,y,x
        return (torch.from_numpy(np.concatenate(all_vf, 0)),
                torch.from_numpy(np.concatenate(all_vc, 0)).int())

    def forward(self, points_list: List[np.ndarray],
                max_voxels: int = MAX_VOXELS) -> Dict[str, torch.Tensor]:
        """
        Args:
            points_list: list of [N_i, 6] numpy arrays (one per batch element)
        Returns:
            dict:
              'bev_tokens'  [B, 196, 2048]
              'seg_logits'  [B, 6, H, W]
              'det_output'  dict
        """
        device = next(self.parameters()).device
        B      = len(points_list)

        # spconv does not support bf16 weights (indice_conv|subm fails).
        # BEV encoder is frozen in Phase 2, so casting to fp32 here is safe
        # and does not interfere with DeepSpeed optimizer state.
        _p = next(self.sparse_encoder.parameters(), None)
        if _p is not None and _p.dtype != torch.float32:
            self.sparse_encoder.float()

        vf, vc = self.batch_voxelize(points_list, max_voxels=max_voxels)
        vf = vf.to(device)
        vc = vc.to(device)

        # Re-apply Native algo each call: load_state_dict / DeepSpeed wrapping can reset
        # self.algo back to MaskImplicitGemm, so we enforce it here to avoid CUDA crashes.
        from spconv.core import ConvAlgo as _ConvAlgo
        for _m in self.sparse_encoder.modules():
            if hasattr(_m, 'algo'):
                _m.algo = _ConvAlgo.Native

        # Skip truly pathological samples. NavSim LiDAR data typically yields 47k–65k
        # voxels per sample after voxelization. vc here is the TOTAL across the whole
        # batch (B samples), so threshold must scale with B to avoid falsely skipping
        # normal batches when batch_size > 1.
        _voxel_threshold = 80000 * B
        if len(vc) > _voxel_threshold:
            import warnings
            warnings.warn(f'[BEVFusionEncoder] skipping high-voxel batch ({len(vc)} voxels, B={B}), returning None')
            return None

        # ── Run sparse_encoder on a dedicated side-stream so main stream stays clean.
        # spconv can deadlock a CUDA kernel for certain bad LiDAR samples. If that
        # happens on the main stream, ALL subsequent CUDA ops (including NCCL ALLREDUCE)
        # queue behind the deadlocked kernel and block all 8 ranks for 10+ minutes.
        # With stream isolation: bev_stream may hang, but main stream never waits for it,
        # so NCCL proceeds normally. The orphaned kernel is eventually killed by the GPU.
        if not hasattr(self, '_bev_stream') or self._bev_stream is None:
            self._bev_stream = torch.cuda.Stream(device=device)

        main_stream = torch.cuda.current_stream(device)
        bev_stream  = self._bev_stream

        # Ensure bev_stream sees the latest vf/vc tensors from the main stream
        bev_stream.wait_stream(main_stream)

        bev = None
        _bev_ok = False
        try:
            with torch.cuda.stream(bev_stream):
                bev = self.sparse_encoder(vf, vc, B)
            _bev_ok = True
        except (ValueError, RuntimeError) as e:
            import warnings
            warnings.warn(f'[BEVFusionEncoder] sparse_encoder failed ({e}), skipping BEV tokens')
            # Discard the corrupted stream and create a fresh one.
            # Without this, the next *successful* call would do main_stream.wait_stream(bev_stream)
            # on a stream that still has a hung kernel, deadlocking the main training stream.
            self._bev_stream = torch.cuda.Stream(device=device)
            return None

        if not _bev_ok or bev is None:
            return None

        # Sparse encoder succeeded — bring results back to main stream
        main_stream.wait_stream(bev_stream)

        # SECOND backbone → list of multi-scale features
        ms = self.backbone(bev)

        # SECONDFPN neck: takes backbone multi-scale outputs directly
        # ms = [blocks.0(128ch, H, W), blocks.1(256ch, H/2, W/2)]
        # neck outputs [256ch, 256ch] → concat → 512ch
        bev_feat = self.neck(ms)[0]   # neck returns a list, take first item

        return {
            'bev_tokens': self.bev_projector(bev_feat),  # [B, 196, 2048]
            'seg_logits': self.seg_head(bev_feat),        # [B, 6, H, W]
            'det_output': self.det_head(bev_feat),        # dict
        }
