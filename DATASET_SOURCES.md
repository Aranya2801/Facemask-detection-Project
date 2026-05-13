# 📦 Dataset Sources — FaceMask Detection System

All datasets listed here are **free and publicly available**.

---

## ✅ Recommended Datasets

### 1. Face Mask Detection (RMFD) — Kaggle ⭐ BEST
- **Images:** 7,553
- **Classes:** With Mask, Without Mask
- **Format:** PASCAL VOC XML annotations
- **Download:** https://www.kaggle.com/andrewmvd/face-mask-detection
- **License:** CC0 Public Domain
- **Notes:** Most widely used, high quality images

### 2. Face Mask 12K — Kaggle ⭐ LARGE
- **Images:** 11,792
- **Classes:** WithMask, WithoutMask
- **Format:** Organized folders
- **Download:** https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
- **License:** CC0
- **Notes:** Excellent for augmenting with dataset 1

### 3. MaskedFace-Net — GitHub ⭐ HUGE
- **Images:** 137,016 (synthetic)
- **Classes:** CMFD (correctly masked), IMFD (incorrectly masked)
- **Download:** https://github.com/cabani/MaskedFace-Net
- **License:** CC BY-NC-SA 4.0
- **Notes:** Best for training incorrect mask class

### 4. COVID-19 Face Mask Dataset — Kaggle
- **Images:** 1,006
- **Classes:** With Mask, Without Mask
- **Download:** https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset

---

## 📥 How to Download

### Via Kaggle CLI (Fastest)
```bash
pip install kaggle

# Setup API key at: https://www.kaggle.com/account → API → Create Token
# Place kaggle.json in ~/.kaggle/

# Download dataset 1
kaggle datasets download -d andrewmvd/face-mask-detection
unzip face-mask-detection.zip -d downloads/

# Download dataset 2
kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset
unzip face-mask-12k-images-dataset.zip -d downloads/
```

### Via Browser
1. Sign into Kaggle
2. Click dataset link above
3. Click "Download All" button
4. Unzip to `downloads/`

---

## 📁 Required Folder Structure

After downloading, organize your images:

```
dataset/
├── with_mask/               ← Put all "with mask" images here
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── without_mask/            ← Put all "without mask" images here
│   ├── img_001.jpg
│   └── ...
└── mask_weared_incorrect/   ← Incorrectly worn mask images
    ├── img_001.jpg
    └── ...
```

---

## 📊 Minimum Recommended Dataset Size

| Size | Accuracy | Training Time (CPU) |
|---|---|---|
| 500/class (1,500 total) | ~90% | ~30 min |
| 1,000/class (3,000 total) | ~95% | ~1 hr |
| 3,000/class (9,000 total) | ~98% | ~3 hrs |
| 5,000+/class | ~99% | ~6+ hrs |

---

## 🤖 Auto-Download Script

```bash
python scripts/download_dataset.py
```

This will guide you through the full download process.

---

## ✅ After Downloading

```bash
# Verify your dataset
python scripts/prepare_dataset.py --verify

# Start training
python src/train.py
```
