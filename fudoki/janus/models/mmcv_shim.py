"""
mmcv_shim.py

Minimal compatibility shim for mmcv 1.x API used by BEVFusion source code.
Provides: mmcv.runner.{auto_fp16, BaseModule}, mmcv.cnn.{build_conv_layer,
build_norm_layer, build_upsample_layer} using PyTorch primitives.

Usage: call install() before importing any bevfusion/mmdet3d modules.
"""

import sys
import types
import torch.nn as nn


def _auto_fp16(apply_to=None, out_fp32=False):
    """No-op decorator (fp16 handled by PyTorch AMP instead)."""
    def decorator(fn):
        return fn
    return decorator


def _build_norm_layer(cfg, num_features):
    t   = (cfg or {}).get('type', 'BN1d')
    eps = (cfg or {}).get('eps', 1e-3)
    mom = (cfg or {}).get('momentum', 0.01)
    if t == 'BN1d':
        layer = nn.BatchNorm1d(num_features, eps=eps, momentum=mom)
    elif t in ('BN', 'BN2d'):
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=mom)
    elif t == 'BN3d':
        layer = nn.BatchNorm3d(num_features, eps=eps, momentum=mom)
    elif t == 'GN':
        num_groups = (cfg or {}).get('num_groups', 32)
        layer = nn.GroupNorm(num_groups, num_features, eps=eps)
    else:
        layer = nn.BatchNorm1d(num_features, eps=eps, momentum=mom)
    return 'bn', layer


def _build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    t = cfg.get('type', 'Conv2d')
    if t == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    if t == 'Conv3d':
        return nn.Conv3d(*args, **kwargs)
    if t == 'Conv1d':
        return nn.Conv1d(*args, **kwargs)
    if t == 'ConvTranspose2d':
        return nn.ConvTranspose2d(*args, **kwargs)
    # SubMConv3d / SparseConv3d delegated externally
    return nn.Conv2d(*args, **kwargs)


def _build_upsample_layer(cfg, *args, **kwargs):
    t = (cfg or {}).get('type', 'deconv')
    if t in ('deconv', 'ConvTranspose2d'):
        return nn.ConvTranspose2d(*args, **kwargs)
    if t == 'pixel_shuffle':
        scale = kwargs.pop('scale_factor', 2)
        return nn.PixelShuffle(scale)
    return nn.ConvTranspose2d(*args, **kwargs)


def install():
    """Inject mmcv 1.x shims into sys.modules."""
    if 'mmcv.runner' in sys.modules and hasattr(sys.modules['mmcv.runner'], 'auto_fp16'):
        return  # already installed

    # mmcv root
    mmcv_mod = sys.modules.get('mmcv') or types.ModuleType('mmcv')
    mmcv_mod.__version__ = '1.6.0'

    # mmcv.runner
    runner = types.ModuleType('mmcv.runner')
    class _BaseModule(nn.Module):
        """Drop-in replacement for mmcv BaseModule that accepts init_cfg."""
        def __init__(self, *args, init_cfg=None, **kwargs):
            super().__init__()

    runner.auto_fp16 = _auto_fp16
    runner.BaseModule = _BaseModule

    # mmcv.cnn
    cnn = types.ModuleType('mmcv.cnn')
    cnn.build_conv_layer   = _build_conv_layer
    cnn.build_norm_layer   = _build_norm_layer
    cnn.build_upsample_layer = _build_upsample_layer

    sys.modules['mmcv']        = mmcv_mod
    sys.modules['mmcv.runner'] = runner
    sys.modules['mmcv.cnn']    = cnn
    mmcv_mod.runner = runner
    mmcv_mod.cnn    = cnn
