"""
Precompute GT BEV seg maps and obstacle heatmaps from GoalFlow trainval cache.

GoalFlow cache covers 102899/103288 JSONL tokens (99.6%).
Coordinate mapping:
  GoalFlow BEV [128, 256] at 0.25m/px: row=x/0.25, col=y/0.25+128 (x∈[0,32], y∈[-32,32])
  CMMI BEV [180, 180] at 0.60m/px: col=(x+54)/0.6, row=(54-y)/0.6 (x,y∈[-54,54])

Run with:
  conda run -n cmmi python scripts/precompute_gt_labels.py [--workers N]

Then run:
  python scripts/update_jsonl_gt_paths.py
"""

import os, sys, json, gzip, pickle, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# ── BEV constants ─────────────────────────────────────────────────────────────
# CMMI BEV
WAM_RANGE = 54.0   # ±54 m
WAM_RES   = 0.6    # m / pixel
WAM_SIZE  = 180

# GoalFlow BEV
GL_W   = 256      # cols
GL_H   = 128      # rows
GL_RES = 0.25     # m / pixel
GL_Y_CENTER = GL_W / 2  # 128: col offset for y=0

# ── Class mapping: GoalFlow (0-6) → Baseline (0-5) ───────────────────────────
# GoalFlow: 0=bg, 1=road, 2=walkways, 3=centerline, 4=static, 5=vehicles, 6=pedestrians
# Baseline: 0=drivable, 1=ped_crossing, 2=walkway, 3=lane_divider, 4=vehicle, 5=background
GL_TO_WAM = np.array([5, 0, 2, 3, 5, 4, 5], dtype=np.uint8)
#                      bg  rd wk  cl  st  veh ped

# ── Precomputed lookup table: WAM pixel → GoalFlow pixel ─────────────────────
# Build once, reuse for every scene
_rows_wam = np.arange(WAM_SIZE, dtype=np.float32)   # [180]
_cols_wam = np.arange(WAM_SIZE, dtype=np.float32)   # [180]
_c_grid, _r_grid = np.meshgrid(_cols_wam, _rows_wam)  # each [180,180]

# BEV pixel → ego metres
_x_ego = _c_grid * WAM_RES - WAM_RANGE  # [180,180]
_y_ego = WAM_RANGE - _r_grid * WAM_RES  # [180,180]

# ego metres → GoalFlow pixel (float)
_r_gl_f = _x_ego / GL_RES              # [180,180]: row in GoalFlow
_c_gl_f = _y_ego / GL_RES + GL_Y_CENTER  # [180,180]: col in GoalFlow

# Validity mask: which WAM pixels fall inside GoalFlow's coverage
_valid = (
    (_r_gl_f >= 0) & (_r_gl_f < GL_H) &
    (_c_gl_f >= 0) & (_c_gl_f < GL_W)
)  # [180,180] bool

# Integer GoalFlow indices (clamped to valid range, only used where _valid=True)
_r_gl_i = np.clip(_r_gl_f.round().astype(np.int32), 0, GL_H - 1)  # [180,180]
_c_gl_i = np.clip(_c_gl_f.round().astype(np.int32), 0, GL_W - 1)  # [180,180]


# ── Gaussian heatmap helpers ──────────────────────────────────────────────────

def _gaussian2d(radius: int) -> np.ndarray:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coords = np.arange(diameter, dtype=np.float32) - radius
    y_v, x_v = np.meshgrid(coords, coords, indexing='ij')
    g = np.exp(-(x_v * x_v + y_v * y_v) / (2 * sigma * sigma)).astype(np.float32)
    g[g < np.finfo(np.float32).eps * g.max()] = 0.0
    return g


def _draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius: int):
    """Place Gaussian at BEV pixel (col=cx, row=cy)."""
    H, W = heatmap.shape
    x, y = int(round(cx)), int(round(cy))
    if not (0 <= x < W and 0 <= y < H):
        return
    g = _gaussian2d(radius)
    left   = min(x, radius);       right  = min(W - x, radius + 1)
    top    = min(y, radius);       bottom = min(H - y, radius + 1)
    hm_roi = heatmap[y - top : y + bottom, x - left : x + right]
    g_roi  = g[radius - top : radius + bottom, radius - left : radius + right]
    if hm_roi.size > 0 and g_roi.shape == hm_roi.shape:
        np.maximum(hm_roi, g_roi, out=hm_roi)


# ── Per-scene worker ──────────────────────────────────────────────────────────

def process_token(args):
    token, cache_path, out_dir = args

    seg_path = os.path.join(out_dir, f"{token}_seg.npy")
    det_path = os.path.join(out_dir, f"{token}_det.npy")
    if os.path.exists(seg_path) and os.path.exists(det_path):
        return token, "skip"

    try:
        with gzip.open(cache_path, 'rb') as f:
            tgt = pickle.load(f)
    except Exception as e:
        return token, f"load_err:{e}"

    # ── 1. Seg map ────────────────────────────────────────────────────────────
    gl_map = tgt['bev_semantic_map']
    if hasattr(gl_map, 'numpy'):
        gl_map = gl_map.numpy()
    gl_map = gl_map.astype(np.uint8)  # [128, 256] values 0-6

    # Default background
    seg = np.full((WAM_SIZE, WAM_SIZE), 5, dtype=np.uint8)

    # Vectorised lookup: sample GoalFlow map at computed indices
    gl_classes = gl_map[_r_gl_i, _c_gl_i]        # [180,180] GoalFlow class 0-6
    seg_classes = GL_TO_WAM[gl_classes]            # [180,180] class 0-5
    seg[_valid] = wam_classes[_valid]

    # ── 2. Det heatmap from agent_states ─────────────────────────────────────
    ag_states = tgt['agent_states']
    ag_labels = tgt['agent_labels']
    if hasattr(ag_states, 'numpy'):
        ag_states = ag_states.numpy()
    if hasattr(ag_labels, 'numpy'):
        ag_labels = ag_labels.numpy()

    heatmap = np.zeros((WAM_SIZE, WAM_SIZE), dtype=np.float32)
    for i in range(len(ag_labels)):
        if not ag_labels[i]:
            continue
        x, y, _h, length, width = float(ag_states[i, 0]), float(ag_states[i, 1]), \
                                   float(ag_states[i, 2]), float(ag_states[i, 3]), \
                                   float(ag_states[i, 4])
        # BEV pixel
        cx = (x + WAM_RANGE) / WAM_RES   # col
        cy = (WAM_RANGE - y) / WAM_RES   # row
        radius = max(1, min(int(max(length, width) / (2 * WAM_RES)), 15))
        _draw_gaussian(heatmap, cx, cy, radius)

    np.save(seg_path, seg)
    np.save(det_path, heatmap.astype(np.float16))
    return token, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",      default="data/navsim_103k_bev.jsonl")
    parser.add_argument("--cache_root", default="./data/cache")
    parser.add_argument("--output_dir", default="data/gt_labels")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--end",        type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build cache index ─────────────────────────────────────────────────────
    print("Building cache index …")
    cache_root = Path(args.cache_root)
    token_to_cache: dict[str, str] = {}
    for log in cache_root.iterdir():
        if not log.is_dir():
            continue
        for tok_dir in log.iterdir():
            gz = tok_dir / "transfuser_target.gz"
            if gz.exists():
                token_to_cache[tok_dir.name] = str(gz)
    print(f"Cache index: {len(token_to_cache)} tokens")

    # ── Load JSONL token list ─────────────────────────────────────────────────
    tokens = []
    with open(args.jsonl) as f:
        for line in f:
            tokens.append(json.loads(line)["id"])
    end = len(tokens) if args.end == -1 else args.end
    tokens = tokens[args.start:end]
    print(f"JSONL slice: {len(tokens)} tokens [{args.start}:{end}]")

    # Filter to cache hits
    tasks = []
    missing = 0
    for tok in tokens:
        if tok in token_to_cache:
            tasks.append((tok, token_to_cache[tok], str(out_dir)))
        else:
            missing += 1
    print(f"Cache hits: {len(tasks)}, not in cache: {missing}")

    # ── Process ───────────────────────────────────────────────────────────────
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_token, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="precompute"):
            tok, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"[WARN] {tok}: {status}")

    print(f"\nDone: {ok} generated, {skip} skipped (already exist), {err} errors, {missing} not in cache.")
    print(f"Run: python scripts/update_jsonl_gt_paths.py")


if __name__ == "__main__":
    main()
