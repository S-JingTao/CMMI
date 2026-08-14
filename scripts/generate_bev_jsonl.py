"""
Generate navsim_668k_bev.jsonl from navsim_668k.jsonl.

For each entry:
  - If LiDAR PCD exists: add 'lidar' field (absolute path) and insert <bev>
    after <image> in the human message.
  - If not: keep entry unchanged (no lidar field, no <bev> tag).

Usage:
    python scripts/generate_bev_jsonl.py \
        --input  data/navsim_668k.jsonl \
        --output data/navsim_668k_bev.jsonl \
        --lidar_root /path/to/dataset/sensor_blobs/trainval
"""

import argparse
import json
import os
import sys

def build_lidar_path(image_path: str, frame_id: str, lidar_root: str) -> str:
    """
    Derive absolute LiDAR PCD path from image path and frame id.

    image_path: 'data/navsim_data/sensor_blobs/trainval/<log>/CAM_F0/<cam_id>.jpg'
    frame_id:   NAVSIM frame token (= PCD filename stem)
    lidar_root: '/path/to/sensor_blobs/trainval'
    """
    parts = image_path.split('/')
    log_name = parts[-3]
    return os.path.join(lidar_root, log_name, 'MergedPointCloud', f'{frame_id}.pcd')


def add_bev_to_conversation(conversations: list) -> list:
    """Insert <bev> after <image> in the first human message."""
    result = []
    for turn in conversations:
        if turn['from'] == 'human' and '<image>' in turn['value']:
            turn = dict(turn)
            turn['value'] = turn['value'].replace('<image>', '<image>\n<bev>', 1)
        result.append(turn)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      default='data/navsim_668k.jsonl')
    parser.add_argument('--output',     default='data/navsim_668k_bev.jsonl')
    parser.add_argument('--lidar_root',
                        default='/path/to/dataset/sensor_blobs/trainval')
    args = parser.parse_args()

    total = 0
    with_bev = 0

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            frame_id  = item.get('id', '')
            img_path  = item.get('image', '')

            if frame_id and img_path:
                pcd_path = build_lidar_path(img_path, frame_id, args.lidar_root)
                if os.path.exists(pcd_path):
                    item['lidar'] = pcd_path
                    item['conversations'] = add_bev_to_conversation(item['conversations'])
                    with_bev += 1

            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

            if total % 50000 == 0:
                print(f'  processed {total:,}  with_bev={with_bev:,}  '
                      f'({100*with_bev/total:.1f}%)', flush=True)

    print(f'Done. total={total:,}  with_bev={with_bev:,}  ({100*with_bev/total:.1f}%)')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
