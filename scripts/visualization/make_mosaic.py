"""
Tile 20 CMMI BEV visualization images into a 5×4 mosaic.
Rows:
  0: Straight — stopped     (v ≤ 0.6 m/s)
  1: Straight — very slow   (v = 1.0–1.4 m/s)
  2: Straight — slow/medium (v = 2.3–5.9 m/s)
  3: Straight — fast        (v ≥ 7.1 m/s)
  4: Turning                (left ×2, right ×2)
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

VIS_DIR = Path("exp/cmmi_bev_vis")
OUT     = Path("exp/cmmi_bev_vis/mosaic_5x4.png")

# ── scene assignments ───────────────────────────────────────────────────────
# (token, cmd, vel) sorted within each row by velocity
rows = [
    {
        "label": "Straight  |  v ≤ 0.6 m/s  (stopped / near-stop)",
        "tokens": [
            "89ae8d041c145f8f",   # v=0.0
            "ade979d99d51517f",   # v=0.0
            "8b788f2715985cee",   # v=0.0
            "2ee8d10b988c58b2",   # v=0.6
        ],
    },
    {
        "label": "Straight  |  v = 1.0–1.4 m/s  (very slow)",
        "tokens": [
            "5ab779b8a0995778",   # v=1.0
            "4b892b0f3efa5255",   # v=1.1
            "ac944958cb6d5209",   # v=1.1
            "b762ea96cfa75157",   # v=1.4
        ],
    },
    {
        "label": "Straight  |  v = 2.3–5.9 m/s  (slow / medium)",
        "tokens": [
            "bc68d211a62d5c19",   # v=2.3
            "dc97241b7037592b",   # v=4.9
            "2c3e04a929b85f30",   # v=5.4
            "2ec48f094e475d85",   # v=5.9
        ],
    },
    {
        "label": "Straight  |  v ≥ 7.1 m/s  (medium / fast)",
        "tokens": [
            "d01d51c26594547c",   # v=7.1
            "e7c485c237b352cd",   # v=7.3
            "c0be0cb86f2c5cf1",   # v=9.4
            "e009079e2dc25bbf",   # v=10.4
        ],
    },
    {
        "label": "Turning  |  Left ×2, Right ×2",
        "tokens": [
            "df84a00692885ec0",   # left  v=3.8
            "64f56e62619850b2",   # left  v=4.8
            "ab94d64d31ea5435",   # right v=4.7
            "c60be3c852c55da2",   # right v=6.1
        ],
    },
]


def find_img(token: str) -> Path:
    matches = list(VIS_DIR.glob(f"{token}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No image for token {token}")
    return matches[0]


def make_mosaic():
    # load all images and get target size (scale to uniform width)
    all_imgs = []
    for row in rows:
        row_imgs = [Image.open(find_img(t)) for t in row["tokens"]]
        all_imgs.append(row_imgs)

    # uniform cell size: max-width / 4 wide, maintain aspect ratio
    sample = all_imgs[0][0]
    cell_w = sample.width
    cell_h = sample.height

    LABEL_H = 44          # row label banner height (px)
    PAD     = 4           # gap between cells

    n_rows, n_cols = 5, 4
    total_w = n_cols * cell_w + (n_cols - 1) * PAD
    total_h = n_rows * (cell_h + LABEL_H) + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    # try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for r_idx, (row_def, row_imgs) in enumerate(zip(rows, all_imgs)):
        y_top = r_idx * (cell_h + LABEL_H + PAD)

        # row label banner
        draw.rectangle(
            [0, y_top, total_w - 1, y_top + LABEL_H - 1],
            fill=(50, 50, 80),
        )
        draw.text((12, y_top + 8), row_def["label"], fill=(220, 220, 255), font=font)

        for c_idx, img in enumerate(row_imgs):
            x = c_idx * (cell_w + PAD)
            y = y_top + LABEL_H
            canvas.paste(img.resize((cell_w, cell_h), Image.LANCZOS), (x, y))

    canvas.save(str(OUT), quality=92)
    print(f"Saved: {OUT}  ({canvas.width}×{canvas.height} px)")


if __name__ == "__main__":
    make_mosaic()
