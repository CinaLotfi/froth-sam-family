import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import tifffile as tiff

from segment_anything.utils.transforms import ResizeLongestSide

# SAM preprocessing (longest side = 1024)
sam_resize = ResizeLongestSide(1024)


class FrothSegmentationDataset(Dataset):
    """
    Dataset for paired .tif[f] images and LabelMe .json annotations.

    Returns for each index:
      - image_raw: H x W x 3, np.uint8 (RGB)
      - mask     : H x W,   np.uint8 in {0,255}
    """
    def __init__(self, root: Path | str, label_key: str = "froth"):
        self.root = Path(root)
        self.label_key = label_key
        self.samples = self._index_pairs()
        if not self.samples:
            raise RuntimeError(f"No paired .tif[f]/.json found in: {self.root}")

    def _index_pairs(self):
        files = [f for f in os.listdir(self.root)
                 if f.lower().endswith((".tif", ".tiff"))]
        pairs = []
        names = set(os.listdir(self.root))
        for f in sorted(files):
            j = f.rsplit(".", 1)[0] + ".json"
            if j in names:
                pairs.append((self.root / f, self.root / j))
        return pairs

    @staticmethod
    def _to_uint8_3ch(img: np.ndarray) -> np.ndarray:
        """
        Convert image to HWC uint8 RGB:
          - handle CHW to HWC
          - grayscale -> 3 channels
          - scale non-uint8 to [0,255] uint8
        """
        arr = np.asarray(img)

        # CHW -> HWC if needed
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        # grayscale -> 3ch
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # ensure 3 channels
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]

        # ensure uint8
        if arr.dtype != np.uint8:
            m = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
            arr = np.clip((arr / m) * 255.0, 0, 255).astype(np.uint8)

        return arr

    def _mask_from_labelme(self, json_path: Path, H: int, W: int) -> np.ndarray:
        """
        Build a single-channel 0/255 mask from LabelMe polygons with label==self.label_key.
        """
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        for shp in shapes:
            if shp.get("label", "") != self.label_key:
                continue
            pts = np.array(shp.get("points", []), dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # load image
        img = tiff.imread(str(img_path))
        img = self._to_uint8_3ch(img)  # HWC uint8, 3ch
        H, W = img.shape[:2]

        # build mask
        mask = self._mask_from_labelme(json_path, H, W)  # HxW uint8 {0,255}

        return img, mask


def preprocess_image_for_sam(image_raw_hwc_uint8: np.ndarray):
    """
    SAM preprocessing with resized-size tracking:
      - Resize longest side to 1024 using SAM's transform
      - Convert to CHW float32 in [0,1]
      - Zero-pad to (3, 1024, 1024)

    Returns:
      img_1024 : (3, 1024, 1024) FloatTensor
      in_size  : (h_resized, w_resized) BEFORE padding
    """
    img_resized = sam_resize.apply_image(image_raw_hwc_uint8)        # H' x W' x 3
    img_t = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    h_resized, w_resized = img_t.shape[1], img_t.shape[2]
    out = torch.zeros((3, 1024, 1024), dtype=img_t.dtype)
    out[:, :h_resized, :w_resized] = img_t
    return out, (h_resized, w_resized)


def collate_froth(batch):
    """
    Collate a list of (image_raw, mask) into a SAM-ready mini-batch.

    Returns dict:
      images_1024: (B,3,1024,1024) float32
      boxes:       (B,4) float32 (original coords [x0,y0,x1,y1])
      orig_sizes:  (B,2) long (H,W)
      input_sizes: (B,2) long (h_resized, w_resized) BEFORE padding
      masks:       list[(H,W) float {0,1}]
    """
    images_1024, boxes, orig_sizes, input_sizes, masks = [], [], [], [], []

    for (img_raw, mask_raw) in batch:
        H, W = mask_raw.shape[:2]

        img_1024, (h_resized, w_resized) = preprocess_image_for_sam(img_raw)

        images_1024.append(img_1024)
        boxes.append(torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32))
        orig_sizes.append(torch.tensor([H, W], dtype=torch.int64))
        input_sizes.append(torch.tensor([h_resized, w_resized], dtype=torch.int64))

        m = torch.from_numpy((mask_raw > 127).astype(np.float32))
        masks.append(m)

    images_1024 = torch.stack(images_1024, dim=0)
    boxes       = torch.stack(boxes, dim=0)
    orig_sizes  = torch.stack(orig_sizes, dim=0)
    input_sizes = torch.stack(input_sizes, dim=0)

    return {
        "images_1024": images_1024,
        "boxes": boxes,
        "orig_sizes": orig_sizes,
        "input_sizes": input_sizes,
        "masks": masks,
    }
