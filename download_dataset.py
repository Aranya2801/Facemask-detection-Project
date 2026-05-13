"""
FaceMask Detection System — Dataset Downloader
================================================
Downloads and organizes multiple public face mask datasets.

Usage:
    python scripts/download_dataset.py                    # All datasets
    python scripts/download_dataset.py --source kaggle    # Kaggle only
    python scripts/download_dataset.py --source github    # GitHub fallback
    python scripts/download_dataset.py --verify           # Verify existing
"""

import os
import sys
import json
import shutil
import argparse
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ─── DATASET REGISTRY ─────────────────────────────────────────────
DATASETS = {
    'rmfd_kaggle': {
        'name':    'Face Mask Detection (Kaggle RMFD)',
        'kaggle':  'andrewmvd/face-mask-detection',
        'images':  '~7,553',
        'classes': ['with_mask', 'without_mask'],
        'notes':   'Most popular face mask dataset. Requires kaggle API key.',
    },
    'facemask_12k': {
        'name':    'Face Mask ~12K (Kaggle)',
        'kaggle':  'ashishjangra27/face-mask-12k-images-dataset',
        'images':  '~11,792',
        'classes': ['with_mask', 'without_mask'],
        'notes':   'Large dataset, good quality.',
    },
    'maskedface_net': {
        'name':    'MaskedFace-Net (GitHub)',
        'url':     'https://github.com/cabani/MaskedFace-Net',
        'images':  '137,016',
        'classes': ['with_mask (correct)', 'with_mask (incorrect)'],
        'notes':   'Synthetic dataset. Manual download required.',
    },
}

# ─── FOLDER STRUCTURE ─────────────────────────────────────────────
CLASS_FOLDERS = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# ─── HELPERS ──────────────────────────────────────────────────────

class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize: self.total = tsize
        self.update(b * bsize - self.n)


def create_dataset_structure(base: str = 'dataset'):
    """Create required folder structure."""
    for cls in CLASS_FOLDERS:
        path = os.path.join(base, cls)
        os.makedirs(path, exist_ok=True)
    print(f"  ✅  Dataset folders created at: {base}/")


def count_images(base: str = 'dataset') -> dict:
    """Count images in each class folder."""
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    counts = {}
    for cls in CLASS_FOLDERS:
        path = os.path.join(base, cls)
        if os.path.exists(path):
            imgs = [f for f in os.listdir(path) if Path(f).suffix.lower() in exts]
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts


def verify_dataset(base: str = 'dataset'):
    """Print dataset summary and check minimum counts."""
    print("\n📊  Dataset Verification")
    print("─" * 50)
    counts = count_images(base)
    total  = sum(counts.values())

    ok = True
    for cls, count in counts.items():
        status = "✅" if count >= 100 else "⚠️ " if count > 0 else "❌"
        if count < 100: ok = False
        print(f"  {status}  {cls:<30} {count:>6} images")

    print("─" * 50)
    print(f"  {'':35} {total:>6} total")

    if total < 300:
        print("\n  ⚠️   Dataset is too small for good training.")
        print("  Please download at least 1,000+ images per class.")
        ok = False
    elif total >= 1000:
        print(f"\n  ✅  Dataset looks good! ({total} images)")
    return ok


def download_kaggle(dataset_slug: str, output_dir: str = 'dataset'):
    """Download a Kaggle dataset using kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("  ❌  kaggle package not found. Install: pip install kaggle")
        print("  Then set up API key: https://www.kaggle.com/docs/api")
        return False

    print(f"  ⬇️   Downloading {dataset_slug} from Kaggle...")
    try:
        os.makedirs('downloads', exist_ok=True)
        import subprocess
        result = subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', dataset_slug,
            '-p', 'downloads/',
            '--unzip'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ❌  Kaggle error: {result.stderr}")
            return False

        print(f"  ✅  Downloaded to downloads/")
        return True
    except Exception as e:
        print(f"  ❌  Failed: {e}")
        return False


def organize_kaggle_rmfd(source: str = 'downloads', dest: str = 'dataset'):
    """Organize RMFD Kaggle dataset into our folder structure."""
    print("\n  📁  Organizing RMFD dataset...")

    # RMFD has: images/ annotations/
    # We use annotations to split into with/without mask
    moved = 0
    for root, dirs, files in os.walk(source):
        for f in files:
            src = os.path.join(root, f)
            ext = Path(f).suffix.lower()
            if ext not in {'.jpg', '.jpeg', '.png'}:
                continue

            # Heuristic from RMFD filename patterns
            fname_lower = f.lower()
            if 'with_mask' in root or 'with_mask' in fname_lower:
                dest_cls = 'with_mask'
            elif 'without_mask' in root or 'without_mask' in fname_lower:
                dest_cls = 'without_mask'
            else:
                continue

            dest_path = os.path.join(dest, dest_cls, f)
            if not os.path.exists(dest_path):
                shutil.copy2(src, dest_path)
                moved += 1

    print(f"  ✅  Organized {moved} images")
    return moved


def print_manual_instructions():
    """Print manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         DATASET DOWNLOAD INSTRUCTIONS                        ║
╚══════════════════════════════════════════════════════════════╝

OPTION 1 — Kaggle CLI (Recommended):
─────────────────────────────────────
1. Create a Kaggle account: https://www.kaggle.com
2. Go to Account → API → Create New Token (downloads kaggle.json)
3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<name>\\.kaggle\\ (Windows)
4. Run:
   pip install kaggle
   kaggle datasets download -d andrewmvd/face-mask-detection
   unzip face-mask-detection.zip -d downloads/

OPTION 2 — Direct Browser Download:
─────────────────────────────────────
1. Face Mask Detection (RMFD):
   https://www.kaggle.com/andrewmvd/face-mask-detection
   → Download → unzip to downloads/

2. Face Mask 12K (3-class):
   https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
   → Download → unzip to downloads/

OPTION 3 — Combined Dataset (Best):
─────────────────────────────────────
Download both datasets above and place images in:
   dataset/with_mask/          ← ~3,000+ images of people with masks
   dataset/without_mask/       ← ~3,000+ images of people without masks
   dataset/mask_weared_incorrect/ ← ~1,000+ images of incorrect mask wearing

MINIMUM RECOMMENDED:
   - 500+ images per class (1,500 total)
   - 2,000+ per class for best accuracy (6,000 total)

AFTER DOWNLOAD:
   python scripts/prepare_dataset.py  ← verify + split
   python src/train.py                ← start training
""")


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='FaceMask Dataset Downloader')
    parser.add_argument('--source', choices=['kaggle', 'github', 'manual', 'all'],
                        default='all')
    parser.add_argument('--dataset-dir', default='dataset')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  📦  FaceMask Detection — Dataset Downloader")
    print("═"*60 + "\n")

    create_dataset_structure(args.dataset_dir)

    if args.verify:
        verify_dataset(args.dataset_dir)
        return

    if args.source in ('kaggle', 'all'):
        print("  Attempting Kaggle download...")
        ok1 = download_kaggle('andrewmvd/face-mask-detection', args.dataset_dir)
        ok2 = download_kaggle('ashishjangra27/face-mask-12k-images-dataset', args.dataset_dir)
        if ok1 or ok2:
            organize_kaggle_rmfd('downloads', args.dataset_dir)
        else:
            print_manual_instructions()

    elif args.source == 'manual':
        print_manual_instructions()

    verify_dataset(args.dataset_dir)
    print("\nNext step:")
    print("  python scripts/prepare_dataset.py")
    print("  python src/train.py\n")


if __name__ == '__main__':
    main()
