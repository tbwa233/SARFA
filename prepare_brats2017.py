"""
prepare_brats2017.py
====================
Converts BraTS2017 NIfTI volumes into 2D .npz slice files that match
the format expected by load_BRATS_data.py

Expected input layout
---------------------
/path/to/BraTS2017/
    train/images/*.nii   (T1 volumes)
    train/masks/*.nii    (label volumes, values 0/1/2/4)
    val/images/*.nii
    val/masks/*.nii
    test/images/*.nii
    test/masks/*.nii

BraTS label convention
-----------------------
  0  = background
  1  = necrotic / non-enhancing tumour core  (NET)
  2  = peritumoral oedema                    (OD)
  4  = GD-enhancing tumour                   (ET)

This code creates three CUMULATIVE binary masks per slice to simulate
ambiguous annotations (three "radiologist opinions"):
  label_0 : NET only            (value 1 > 0)
  label_1 : NET + OD            (values 1,2 > 0)
  label_2 : NET + OD + ET       (values 1,2,4 > 0)  <- full tumour

A slice is kept only if the full-tumour mask is non-empty.

Output layout
-------------
data/BraTS2017/
    train/  <case_id>_slice<zzz>.npz
    val/    ...
    test/   ...

Each .npz contains:
    image  : float32 array [H, W]   - normalized T1 slice (0-255)
    label  : uint8  array [3, H, W] - three binary masks (one per label)
"""

import re
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SRC = "/path/to/BraTS2017"  # root of your BraTS2017 dataset
DST = "data/BraTS2017"      # where processed .npz files go
MIN_TUMOUR_FRACTION = 0.0   # skip slices where the full-tumour
                            # mask covers less than this fraction
                            # of pixels (0 = keep all non-empty)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(volume: np.ndarray) -> np.ndarray:
    """Min-max normalise a volume to [0, 255] float32."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)

def build_cumulative_labels(seg: np.ndarray) -> np.ndarray:
    """
    Given a 2-D segmentation slice with BraTS integer labels {0,1,2,4},
    return a [3, H, W] uint8 array of cumulative binary masks:
        [0]  NET only      (seg == 1)
        [1]  NET + OD      (seg in {1, 2})
        [2]  NET + OD + ET (seg in {1, 2, 4})
    """
    net    = (seg == 1).astype(np.uint8)
    net_od = ((seg == 1) | (seg == 2)).astype(np.uint8)
    full   = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.uint8)
    return np.stack([net, net_od, full], axis=0)  # [3, H, W]

def match_mask_for_image(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """
    Find the mask file whose stem matches the image stem.
    Handles common naming conventions, e.g.:
        image: BraTS17_TCIA_105_1_t1.nii  -> mask: BraTS17_TCIA_105_1_seg.nii
        image: case_00001.nii              -> mask: case_00001.nii
    """
    stem = image_path.stem.replace(".nii", "")  # handle double extension

    # Try direct name match first
    for suffix in ["", "_seg", "_mask"]:
        for ext in [".nii", ".nii.gz"]:
            candidate = mask_dir / (stem + suffix + ext)
            if candidate.exists():
                return candidate

    # Try replacing modality tag (e.g. _t1 -> _seg)
    modality_pattern = re.compile(r"_(t1|t1ce|t2|flair)$", re.IGNORECASE)
    base = modality_pattern.sub("", stem)
    for suffix in ["_seg", "_mask", ""]:
        for ext in [".nii", ".nii.gz"]:
            candidate = mask_dir / (base + suffix + ext)
            if candidate.exists():
                return candidate

    return None

def process_split(
    image_dir: Path,
    mask_dir: Path,
    out_dir: Path,
    min_tumour_fraction: float = 0.0,
) -> int:
    """
    Process one data split (train / val / test).
    Returns the number of slices saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix in {".nii", ".gz"}
    )

    if not image_paths:
        print(f"  [WARNING] No .nii files found in {image_dir}")
        return 0

    total_slices  = 0
    skipped_cases = 0

    for img_path in tqdm(image_paths, desc=f"  {image_dir.parent.name}"):
        mask_path = match_mask_for_image(img_path, mask_dir)
        if mask_path is None:
            print(f"  [WARNING] No mask found for {img_path.name} — skipping")
            skipped_cases += 1
            continue

        img_vol  = nib.load(str(img_path)).get_fdata().astype(np.float32)
        mask_vol = nib.load(str(mask_path)).get_fdata().astype(np.int32)

        img_vol  = normalise(img_vol)
        n_slices = img_vol.shape[-1]          # slice along last (axial) axis
        case_id  = img_path.stem.replace(".nii", "")
        saved    = 0

        for z in range(n_slices):
            img_slice  = img_vol[..., z]   # [H, W]
            mask_slice = mask_vol[..., z]  # [H, W]

            labels = build_cumulative_labels(mask_slice)  # [3, H, W]

            # Skip slices with no tumour
            if labels[2].sum() == 0:
                continue

            if min_tumour_fraction > 0:
                h, w = img_slice.shape
                if labels[2].sum() / (h * w) < min_tumour_fraction:
                    continue

            np.savez_compressed(
                out_dir / f"{case_id}_slice{z:03d}.npz",
                image=img_slice,  # float32 [H, W]
                label=labels,     # uint8   [3, H, W]
            )
            saved += 1

        total_slices += saved

    if skipped_cases:
        print(f"  Skipped {skipped_cases} case(s) with no matching mask.")

    return total_slices

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

src = Path(SRC)
dst = Path(DST)

print(f"BraTS2017 preprocessing")
print(f"  Source : {src}")
print(f"  Output : {dst}")
print(f"  Min tumour fraction : {MIN_TUMOUR_FRACTION}\n")

total = 0
for split in ["train", "val", "test"]:
    image_dir = src / split / "images"
    mask_dir  = src / split / "masks"

    if not image_dir.exists():
        print(f"[INFO] {image_dir} not found — skipping split '{split}'")
        continue

    print(f"Processing split: {split}")
    n = process_split(image_dir, mask_dir, dst / split, MIN_TUMOUR_FRACTION)
    print(f"  -> {n} slices saved to {dst / split}\n")
    total += n

print(f"Done. Total slices saved: {total}")
print(f"All files written to: {dst.resolve()}")