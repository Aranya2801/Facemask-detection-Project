# 📋 Daily Use Guide — FaceMask Detection System v2.0

This guide covers everything you need to run S.I.M.R.A.N. as part of your daily operations.

---

## 🚀 Quick Start (Every Day)

```bash
# Activate environment
conda activate facemask

# Option A: Full dashboard (recommended)
python web/app.py
# → Open http://localhost:5000

# Option B: Standalone webcam window
python detect_webcam.py

# Option C: Entrance gate mode (logs everyone)
python scripts/entrance_mode.py --log --alert-sound
```

---

## 🖥️ Webcam Mode Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `A` | Toggle audio alerts |
| `F` | Toggle fullscreen |
| `R` | Reset session stats |
| `H` | Show help overlay |

---

## 📊 Dashboard Panels

### Live Feed
- Annotated camera stream at 30+ FPS
- Green box = mask on, Red box = no mask, Yellow = incorrect

### Compliance Rate Bar
- Real-time percentage of compliant faces
- Green ≥ 80%, Yellow 50–79%, Red < 50%

### Stats Cards
- With Mask / No Mask / Incorrect / Alerts — live counts

### Compliance Trend Chart
- Last 60 seconds of compliance rate
- Dashed lines at 80% and 50% thresholds

### Event Log
- Every detection event with timestamp and confidence
- Export as CSV anytime

---

## 📁 Where Are My Files?

| File | Location |
|---|---|
| Detection CSV log | `logs/detections.csv` |
| Alert JSON log | `logs/alerts.json` |
| Screenshots | `logs/screenshots/` |
| Trained model | `models/facemask_model.h5` |
| Training curves | `logs/training/<run-id>/` |

---

## 🔔 Alert Configuration

Edit `configs/train_config.yaml`:

```yaml
alerts:
  sound: true           # beep on violation
  screenshot: true      # save image on violation
  cooldown_seconds: 3   # min seconds between alerts
  webhook: ""           # POST to URL on violation
```

---

## 🧠 Retraining the Model

When you have new data:
```bash
# Add new images to dataset/
python scripts/prepare_dataset.py --verify
python src/train.py --epochs 20
# Done — model auto-saved to models/facemask_model.h5
```

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| Camera not opening | Try `--camera 1` or `--camera 2` |
| Low FPS | Reduce resolution: `--width 640 --height 480` |
| Model not found | Run `python src/train.py` first |
| Stream not loading | Check `python web/app.py` is running |
| Audio not working | Install pygame: `pip install pygame` |

---

## 📡 API Cheat Sheet

```bash
# Check status
curl http://localhost:5000/api/status

# Get live stats
curl http://localhost:5000/api/stats

# Detect masks in an image
curl -X POST http://localhost:5000/api/detect -F "image=@photo.jpg"

# Get event log
curl http://localhost:5000/api/logs

# Toggle alerts
curl -X PUT http://localhost:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"alerts_enabled": false}'
```
