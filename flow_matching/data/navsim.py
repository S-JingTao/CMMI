# LINT_ME
import os
import json
import random
import struct

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from fudoki.janus.models import VLChatProcessor


from torch.utils.data.dataloader import default_collate

VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576
BEV_LEN = 196


def navsim_collate_fn(batch):
    """
    Custom collate for SupervisedDataset.
    'bev_points' is kept as a plain list (variable-length np.ndarray or None).
    'gt_seg' / 'gt_det' / 'bev_seg_probs' / 'bev_det_probs' are stacked when all present.
    Everything else uses PyTorch's default_collate.
    """
    bev_points    = [item.pop('bev_points')    for item in batch]
    gt_seg_list   = [item.pop('gt_seg')        for item in batch]
    gt_det_list   = [item.pop('gt_det')        for item in batch]
    bev_seg_list  = [item.pop('bev_seg_probs') for item in batch]
    bev_det_list  = [item.pop('bev_det_probs') for item in batch]
    collated = default_collate(batch)
    collated['bev_points'] = bev_points
    # Stack GT tensors if all items have them; otherwise pass as list
    if all(x is not None for x in gt_seg_list):
        collated['gt_seg'] = torch.stack(gt_seg_list, dim=0)   # [B,180,180]
    else:
        collated['gt_seg'] = None
    if all(x is not None for x in gt_det_list):
        collated['gt_det'] = torch.stack(gt_det_list, dim=0)   # [B,180,180]
    else:
        collated['gt_det'] = None
    # Precomputed BEV features: stack only when ALL samples have cache hit
    if all(x is not None for x in bev_seg_list):
        collated['bev_seg_probs'] = torch.stack(bev_seg_list)   # [B,6,180,180]
        collated['bev_det_probs'] = torch.stack(bev_det_list)   # [B,1,180,180]
    else:
        collated['bev_seg_probs'] = None
        collated['bev_det_probs'] = None
    return collated


def load_pcd_points(pcd_path: str) -> np.ndarray:
    """
    Load a binary .pcd file and return raw points as [N, 4] float32 (x,y,z,intensity).
    Supports both ASCII and binary PCD formats.
    """
    with open(pcd_path, 'rb') as f:
        content = f.read()

    # Parse PCD header
    lines = []
    header_end = 0
    pos = 0
    while pos < len(content):
        end = content.find(b'\n', pos)
        if end == -1:
            break
        line = content[pos:end].decode('utf-8', errors='ignore').strip()
        lines.append(line)
        pos = end + 1
        if line.startswith('DATA'):
            header_end = pos
            break

    fields = []
    data_type = 'ascii'
    n_points = 0
    field_sizes = []
    field_types = []
    for line in lines:
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            field_sizes = [int(x) for x in line.split()[1:]]
        elif line.startswith('TYPE'):
            field_types = line.split()[1:]
        elif line.startswith('POINTS'):
            n_points = int(line.split()[1])
        elif line.startswith('DATA'):
            data_type = line.split()[1]

    if data_type == 'binary':
        point_step = sum(field_sizes) if field_sizes else 22
        raw = np.frombuffer(content[header_end:header_end + n_points * point_step],
                            dtype=np.uint8).reshape(n_points, point_step)
        cols = []
        offset = 0
        for fname, fsize, ftype in zip(fields, field_sizes, field_types):
            dtype = {'F': np.float32, 'I': np.int32, 'U': np.uint32}.get(ftype, np.float32)
            if fsize == 4:
                col = raw[:, offset:offset + 4].copy().view(np.float32 if ftype == 'F' else np.int32).reshape(n_points)
            elif fsize == 2:
                col = raw[:, offset:offset + 2].copy().view(np.uint16).reshape(n_points).astype(np.float32)
            elif fsize == 1:
                col = raw[:, offset:offset + 1].copy().reshape(n_points).astype(np.float32)
            else:
                col = np.zeros(n_points, dtype=np.float32)
            cols.append(col)
            offset += fsize
        points = np.stack(cols, axis=1).astype(np.float32)
    else:
        data_str = content[header_end:].decode('utf-8', errors='ignore')
        rows = []
        for line in data_str.strip().splitlines():
            vals = line.strip().split()
            if len(vals) >= 4:
                rows.append([float(v) for v in vals])
        points = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(fields)), dtype=np.float32)

    # Keep only x, y, z, intensity (first 4 columns)
    if points.shape[1] >= 4:
        return points[:, :4]
    return np.zeros((0, 4), dtype=np.float32)


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


class SupervisedDataset(Dataset):
    """
    Dataset for supervised training on image-text conversation data.

    Loads image-text conversation samples from JSON/JSONL files, processes images (resize/pad/normalize),
    tokenizes text prompts with image placeholders, and formats data for training.
    """
    def __init__(
        self,
        data_list: list,
        vl_chat_processor: VLChatProcessor,
        txt_max_length=500,
        bev_cache_dir: str = '',
    ):
        super().__init__()
        self.vl_chat_processor = vl_chat_processor
        self.txt_max_length = txt_max_length
        self.bev_cache_dir = bev_cache_dir
        self.list_data_dict = []

        self.split_token = None

        self.transform_img = transforms.Compose([
            transforms.Lambda(resize_pad),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        for path in data_list:
            jsonl_list = []
            ext = os.path.splitext(path)[-1].lower()

            with open(path, "r", encoding="utf-8") as f:
                if ext == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            jsonl_list.append(json.loads(line))
                elif ext == ".json":
                    data = json.load(f)
                    if isinstance(data, list):
                        jsonl_list.extend(data)
                    else:
                        jsonl_list.append(data)
                else:
                    raise ValueError(f"Unsupported file extension: {ext}")

            self.list_data_dict.extend(jsonl_list)


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            print(f"[Warning] Error loading index {i}: {e}")
            # random choice
            rand_idx = np.random.randint(0, len(self.list_data_dict))
            print(f"[Retry] Loading random index {rand_idx} instead.")
            return self.__getitem__(rand_idx)


    def _get_item(self, i):
        item = self.list_data_dict[i]

        if "image" not in item:
            raise ValueError("Currently only image-based samples are supported.")

        image_list = item["image"]
        lidar_path = item.get("lidar", None)
        conversation = item["conversations"]

        if conversation[0]["from"] == "system":
            addition_system_prompt = conversation[0]["value"]
            conversation = conversation[1:]
        else:
            addition_system_prompt = ""

        conversation = self.construct_conv(conversation)

        # Load GT seg / det from pre-computed .npy paths (if present in jsonl)
        gt_seg_path = item.get("gt_seg", None)
        gt_det_path = item.get("gt_det", None)
        gt_seg = None
        gt_det = None
        if gt_seg_path and os.path.exists(gt_seg_path):
            gt_seg = torch.from_numpy(np.load(gt_seg_path).astype(np.int64))
        if gt_det_path and os.path.exists(gt_det_path):
            gt_det = torch.from_numpy(np.load(gt_det_path).astype(np.float32))

        # Load precomputed BEV features (seg_probs + det_probs) from cache
        # These are saved by precompute_bev_features.py as {token_id}.npz
        bev_seg_probs = None
        bev_det_probs = None
        if self.bev_cache_dir:
            token_id   = item.get('id', '')
            cache_path = os.path.join(self.bev_cache_dir, f'{token_id}.npz')
            if token_id and os.path.exists(cache_path):
                cache = np.load(cache_path)
                bev_seg_probs = torch.from_numpy(cache['seg_probs'].astype(np.float16))  # [6,180,180]
                bev_det_probs = torch.from_numpy(cache['det_probs'].astype(np.float16))  # [1,180,180]

        data_dict = self.process_image_item(
            image_list,
            conversation,
            system_prompt=addition_system_prompt,
            txt_max_length=self.txt_max_length,
            lidar_path=lidar_path,
            gt_seg=gt_seg,
            gt_det=gt_det,
            bev_seg_probs=bev_seg_probs,
            bev_det_probs=bev_det_probs,
        )
        return data_dict

    def construct_conv(self, conv):
        _conv = []
        for item in conv:
            role = item["from"]
            content = item["value"]
            _item = {}

            if role == "human":
                _item["role"] = "User"
            elif role == "gpt":
                _item["role"] = "Assistant"
            else:
                raise ValueError("role must be human or gpt")

            if "<image>" in content:
                content = content.replace("<image>", "<image_placeholder>")
            if "<bev>" in content:
                content = content.replace("<bev>", "<bev_placeholder>")

            _item["content"] = content

            _conv.append(_item)

        return _conv

    def _find_split_token(self, input_ids, split_token_length):
        # start index for "Assistant:"
        start_index = -1
        for j in range(len(input_ids) - split_token_length, 0, -1):
            if input_ids[j:j + split_token_length].numpy().tolist() == self.split_token:
                start_index = j
                break
        return start_index

    def process_image_item(
        self,
        image_paths,
        conversation,
        system_prompt="",
        txt_max_length=500,
        lidar_path=None,
        gt_seg=None,
        gt_det=None,
        bev_seg_probs=None,
        bev_det_probs=None,
    ):
        imgs = []
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            imgs.append(self.transform_img(img))

        if len(imgs) > 0:
            imgs = torch.stack(imgs, dim=0)   # [N, C, H, W]
            img_len = len(imgs) * IMG_LEN
        else:
            imgs = None
            img_len = 0

        # Load LiDAR point cloud if provided
        has_bev = lidar_path is not None and os.path.exists(lidar_path)
        bev_points = None
        bev_len = BEV_LEN if has_bev else 0
        if has_bev:
            bev_points = load_pcd_points(lidar_path)  # [N, 4]

        # gt_seg [180,180] int64 and gt_det [180,180] float32 passed in from _get_item

        generation_understanding_indicator = 0

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt + system_prompt,
        )

        # tokenize
        input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask = input_ids == self.vl_chat_processor.image_id
        image_indices = image_token_mask.nonzero()
        assert len(image_indices) == len(image_paths), \
            f"Number of images ({len(image_paths)}) does not match number of image tokens ({len(image_indices)})"

        input_ids, _ = self.vl_chat_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # add bev tokens to the input_ids
        bev_id = self.vl_chat_processor.bev_id
        if bev_id is not None:
            bev_single_mask = input_ids == bev_id
            bev_single_indices = bev_single_mask.nonzero()
            input_ids = self.vl_chat_processor.add_bev_token(
                bev_indices=bev_single_indices,
                input_ids=input_ids,
            )

        # pad tokens
        total_special_len = img_len + bev_len
        if input_ids.shape[0] >= txt_max_length + total_special_len:
            rows_to_pad = random.randint(0, 50)
        else:
            rows_to_pad = txt_max_length + total_special_len - input_ids.shape[0]
        input_ids = torch.cat([input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
        attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
        attention_mask[:] = True

        # obtain image token mask and zero out image positions in input_ids
        if imgs is not None:
            image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
            image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
            input_ids[image_expanded_mask_indices] = 0
        else:
            image_expanded_token_mask = torch.zeros_like(input_ids)

        # obtain bev token mask and zero out bev positions in input_ids
        if has_bev and bev_id is not None:
            bev_expanded_token_mask = (input_ids == bev_id).to(dtype=int)
            bev_expanded_mask_indices = torch.where(bev_expanded_token_mask == 1)[0]
            input_ids[bev_expanded_mask_indices] = 0
        else:
            bev_expanded_token_mask = torch.zeros_like(input_ids)

        # obtain text token mask
        if self.split_token is None:
            self.split_token = self.vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_token_length = len(self.split_token)
        start_index = self._find_split_token(input_ids, split_token_length)

        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        if start_index != -1:
            text_expanded_token_mask[(start_index + split_token_length):] = 1
        else:
            raise ValueError("Split token not found in input_ids")

        generation_or_understanding_mask = generation_understanding_indicator
        data_info = {}
        data_info['text_token_mask'] = text_expanded_token_mask
        data_info['image_token_mask'] = image_expanded_token_mask
        data_info['bev_token_mask'] = bev_expanded_token_mask
        data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask])

        data_info['attention_mask'] = attention_mask
        data_info['sft_format'] = sft_format

        data_info['understanding_img'] = imgs
        data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int)
        data_info['has_bev'] = torch.Tensor([1 if has_bev else 0]).to(dtype=int)
        data_info['bev_points']    = bev_points     # np.ndarray [N,4] or None
        data_info['gt_seg']        = gt_seg         # [180,180] int64 or None
        data_info['gt_det']        = gt_det         # [180,180] float32 or None
        data_info['bev_seg_probs'] = bev_seg_probs  # [6,180,180] fp16 or None (cached)
        data_info['bev_det_probs'] = bev_det_probs  # [1,180,180] fp16 or None (cached)

        data_info["input_ids"] = torch.LongTensor(input_ids)

        return data_info
