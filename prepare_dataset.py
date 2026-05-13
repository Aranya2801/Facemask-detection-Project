"""
FaceMask Detection System — Dataset Preparation Script
=======================================================
Verifies, cleans, and splits dataset into train/val/test sets.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --split 0.8 --source dataset/
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
CLASS_FOLDERS = ['with_mask', 'without_mask', 'mask_weared_incorrect']


def verify_image(path: str) -> bool:
    """Check if image can be opened."""
    try:
        import cv2
        img = cv2.imread(path)
        return img is not None and img.size > 0
    except Exception:
        return False


def prepare_dataset(source: str = 'dataset', split: float = 0.8, verify: bool = False):
    print("\n" + "═"*60)
    print("  🗂️   Dataset Preparation")
    print("═"*60 + "\n")

    stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'corrupt': 0})

    for cls in CLASS_FOLDERS:
        cls_path = os.path.join(source, cls)
        if not os.path.exists(cls_path):
            print(f"  ⚠️   Missing: {cls_path}")
            continue

        files = [f for f in os.listdir(cls_path)
                 if Path(f).suffix.lower() in VALID_EXTS]
        stats[cls]['total'] = len(files)

        if verify:
            print(f"  🔍  Verifying {cls} ({len(files)} files)...")
            for f in files:
                fpath = os.path.join(cls_path, f)
                if verify_image(fpath):
                    stats[cls]['valid'] += 1
                else:
                    stats[cls]['corrupt'] += 1
                    print(f"    ❌  Corrupt: {f}")
        else:
            stats[cls]['valid'] = len(files)

    # Summary
    print("\n  CLASS SUMMARY:")
    print("  " + "─"*48)
    total = 0
    for cls in CLASS_FOLDERS:
        n = stats[cls]['valid']
        total += n
        bar = '█' * min(int(n / 100), 30)
        print(f"  {cls:<35} {n:>5}  {bar}")
    print("  " + "─"*48)
    print(f"  {'TOTAL':<35} {total:>5}")

    if total < 300:
        print("\n  ❌  Dataset too small. Download more images.")
        print("  Run: python scripts/download_dataset.py")
        return False

    print(f"\n  ✅  Dataset looks good! ({total} images)")
    print(f"\n  Train split:      {int(total * split)} images")
    print(f"  Validation split: {int(total * (1 - split))} images\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='dataset')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--verify', action='store_true',
                        help='Verify each image can be opened (slow)')
    args = parser.parse_args()
    prepare_dataset(args.source, args.split, args.verify)
