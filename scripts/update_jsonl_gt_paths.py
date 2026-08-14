"""
After precompute_gt_labels.py finishes, run this to add gt_seg / gt_det
path fields into the jsonl.

Usage: python scripts/update_jsonl_gt_paths.py
"""
import json
from pathlib import Path

JSONL    = Path("data/navsim_103k_bev.jsonl")
GT_DIR   = Path("data/gt_labels")
OUT_JSONL= Path("data/navsim_103k_bev_gt.jsonl")

found = missing = 0
with open(JSONL) as fin, open(OUT_JSONL, "w") as fout:
    for line in fin:
        item = json.loads(line)
        token = item["id"]
        seg_p = GT_DIR / f"{token}_seg.npy"
        det_p = GT_DIR / f"{token}_det.npy"
        if seg_p.exists() and det_p.exists():
            item["gt_seg"] = str(seg_p)
            item["gt_det"] = str(det_p)
            found += 1
        else:
            missing += 1
        fout.write(json.dumps(item) + "\n")

print(f"Done: {found} with GT, {missing} without GT → {OUT_JSONL}")
