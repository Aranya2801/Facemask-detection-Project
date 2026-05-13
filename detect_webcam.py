"""
╔══════════════════════════════════════════════════════════════╗
║          FaceMask Detection System — Real-Time Webcam        ║
║          Author: Aranya2801 | MIT License | v2.0             ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python detect_webcam.py
    python detect_webcam.py --camera 1 --threshold 0.5 --no-alerts
    python detect_webcam.py --fullscreen --log --save-alerts
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import threading
from datetime import datetime
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import FaceMaskDetector
from alert_engine import AlertEngine
from utils import draw_detections, calculate_compliance_rate, setup_logger

# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-Time Face Mask Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls:
  Q / ESC   - Quit
  S         - Screenshot
  A         - Toggle alerts
  F         - Toggle fullscreen
  L         - Toggle logs panel
  R         - Reset statistics
  H         - Show help overlay
        """
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--model', type=str,
                        default='models/facemask_model.h5',
                        help='Path to trained model')
    parser.add_argument('--face-model', type=str,
                        default='models/face_detector.caffemodel',
                        help='Path to OpenCV face detector model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--face-conf', type=float, default=0.5,
                        help='Face detection confidence (default: 0.5)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Frame height (default: 720)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Start in fullscreen mode')
    parser.add_argument('--no-alerts', action='store_true',
                        help='Disable audio alerts')
    parser.add_argument('--log', action='store_true',
                        help='Enable event logging to CSV')
    parser.add_argument('--save-alerts', action='store_true',
                        help='Save screenshots on non-compliance')
    parser.add_argument('--entrance-mode', action='store_true',
                        help='Enable entrance gate mode (count + log)')
    parser.add_argument('--show-fps', action='store_true', default=True,
                        help='Show FPS counter')
    parser.add_argument('--show-confidence', action='store_true', default=True,
                        help='Show confidence scores on detections')
    return parser.parse_args()


# ─── STATS TRACKER ────────────────────────────────────────────────────────────
class SessionStats:
    """Track real-time statistics for the current session."""

    def __init__(self, history_len=300):
        self.session_start = datetime.now()
        self.total_detections = 0
        self.with_mask_count = 0
        self.without_mask_count = 0
        self.incorrect_mask_count = 0
        self.alert_count = 0
        self.screenshot_count = 0
        # Rolling history for trend line (last 300 frames)
        self.compliance_history = deque(maxlen=history_len)
        self.fps_history = deque(maxlen=30)
        self.lock = threading.Lock()

    def update(self, detections):
        with self.lock:
            for d in detections:
                self.total_detections += 1
                label = d.get('label', '')
                if label == 'with_mask':
                    self.with_mask_count += 1
                elif label == 'without_mask':
                    self.without_mask_count += 1
                    self.alert_count += 1
                elif label == 'mask_incorrect':
                    self.incorrect_mask_count += 1

            rate = calculate_compliance_rate(
                self.with_mask_count,
                self.without_mask_count,
                self.incorrect_mask_count
            )
            self.compliance_history.append(rate)

    def get_compliance_rate(self):
        total = self.with_mask_count + self.without_mask_count + self.incorrect_mask_count
        if total == 0:
            return 100.0
        return (self.with_mask_count / total) * 100.0

    def get_uptime(self):
        delta = datetime.now() - self.session_start
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def reset(self):
        with self.lock:
            self.total_detections = 0
            self.with_mask_count = 0
            self.without_mask_count = 0
            self.incorrect_mask_count = 0
            self.alert_count = 0
            self.compliance_history.clear()


# ─── OVERLAY RENDERER ─────────────────────────────────────────────────────────
class UIOverlay:
    """Render HUD / stats overlay on frame."""

    # Color scheme (BGR)
    GREEN  = (0, 210, 100)
    RED    = (50, 60, 220)
    YELLOW = (0, 180, 240)
    WHITE  = (240, 240, 240)
    BLACK  = (20, 20, 20)
    TEAL   = (180, 210, 0)
    DARK   = (15, 22, 35)

    def __init__(self):
        self.show_logs = True
        self.show_help = False
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.mono = cv2.FONT_HERSHEY_PLAIN

    def draw_panel(self, frame, x, y, w, h, alpha=0.75):
        """Draw semi-transparent dark panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.DARK, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Border
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.TEAL, 1)

    def draw_stats_panel(self, frame, stats, fps):
        """Top-right stats panel."""
        fw, fh = frame.shape[1], frame.shape[0]
        pw, ph = 280, 200
        px = fw - pw - 10
        py = 10

        self.draw_panel(frame, px, py, pw, ph)

        rate = stats.get_compliance_rate()
        color = self.GREEN if rate >= 80 else self.YELLOW if rate >= 50 else self.RED

        # Title
        cv2.putText(frame, "SYSTEM STATUS", (px + 10, py + 22),
                    self.font, 0.45, self.TEAL, 1, cv2.LINE_AA)
        cv2.line(frame, (px + 10, py + 28), (px + pw - 10, py + 28), self.TEAL, 1)

        lines = [
            (f"FPS:          {fps:.1f}", self.WHITE),
            (f"Uptime:       {stats.get_uptime()}", self.WHITE),
            (f"Compliance:   {rate:.1f}%", color),
            (f"With Mask:    {stats.with_mask_count}", self.GREEN),
            (f"No Mask:      {stats.without_mask_count}", self.RED),
            (f"Incorrect:    {stats.incorrect_mask_count}", self.YELLOW),
            (f"Alerts:       {stats.alert_count}", self.RED),
        ]

        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (px + 10, py + 50 + i * 22),
                        self.mono, 1.1, col, 1, cv2.LINE_AA)

    def draw_compliance_bar(self, frame, stats):
        """Bottom compliance bar."""
        fw = frame.shape[1]
        fh = frame.shape[0]
        rate = stats.get_compliance_rate()

        bw = fw - 40
        bh = 18
        bx = 20
        by = fh - 40

        # Background
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 35, 45), -1)
        # Fill
        fill_w = int(bw * rate / 100)
        color = self.GREEN if rate >= 80 else self.YELLOW if rate >= 50 else self.RED
        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bh), color, -1)
        # Border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.TEAL, 1)
        # Label
        cv2.putText(frame, f"COMPLIANCE RATE: {rate:.1f}%",
                    (bx + 8, by + 13), self.mono, 1.0, self.WHITE, 1, cv2.LINE_AA)

    def draw_title_bar(self, frame, alerts_on):
        """Top-left title bar."""
        cv2.rectangle(frame, (10, 10), (380, 60), self.DARK, -1)
        cv2.rectangle(frame, (10, 10), (380, 60), self.TEAL, 1)

        cv2.putText(frame, "FACEMASK DETECTION SYSTEM",
                    (18, 32), self.font, 0.55, self.TEAL, 1, cv2.LINE_AA)
        cv2.putText(frame, f"v2.0  |  Alerts: {'ON' if alerts_on else 'OFF'}",
                    (18, 52), self.mono, 1.0,
                    self.GREEN if alerts_on else self.RED, 1, cv2.LINE_AA)

    def draw_help(self, frame):
        """Help overlay."""
        fw, fh = frame.shape[1], frame.shape[0]
        self.draw_panel(frame, fw//2-200, fh//2-160, 400, 320, alpha=0.92)
        lines = [
            ("KEYBOARD CONTROLS", self.TEAL),
            ("", self.WHITE),
            ("Q / ESC  Quit", self.WHITE),
            ("S        Screenshot", self.WHITE),
            ("A        Toggle Alerts", self.WHITE),
            ("F        Fullscreen", self.WHITE),
            ("L        Toggle Log Panel", self.WHITE),
            ("R        Reset Statistics", self.WHITE),
            ("H        Toggle this Help", self.WHITE),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (fw//2 - 180, fh//2 - 140 + i*30),
                        self.font, 0.5, col, 1, cv2.LINE_AA)

    def draw_detection_boxes(self, frame, detections, show_confidence=True):
        """Draw bounding boxes and labels for detections."""
        colors = {
            'with_mask':    self.GREEN,
            'without_mask': self.RED,
            'mask_incorrect': self.YELLOW,
        }
        labels_text = {
            'with_mask':    'MASK ON',
            'without_mask': 'NO MASK',
            'mask_incorrect': 'INCORRECT',
        }

        for det in detections:
            x, y, w, h = det['bbox']
            label = det.get('label', 'unknown')
            conf  = det.get('confidence', 0.0)
            color = colors.get(label, self.WHITE)
            text  = labels_text.get(label, label.upper())
            if show_confidence:
                text += f" {conf:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Corner accents
            corner_len = 15
            corner_thick = 3
            pts = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
            dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
            for (cx,cy),(dx,dy) in zip(pts,dirs):
                cv2.line(frame,(cx,cy),(cx+dx*corner_len,cy),color,corner_thick)
                cv2.line(frame,(cx,cy),(cx,cy+dy*corner_len),color,corner_thick)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 5),
                        self.font, 0.55, self.BLACK, 1, cv2.LINE_AA)


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    logger = setup_logger('webcam_detect', 'logs/webcam.log' if args.log else None)
    logger.info("Starting FaceMask Detection System v2.0")

    # Initialize detector
    print("\n" + "═"*60)
    print("  🛡️  FaceMask Detection System v2.0")
    print("═"*60)
    print(f"  📷  Camera: {args.camera}")
    print(f"  🧠  Model:  {args.model}")
    print(f"  📏  Resolution: {args.width}x{args.height}")
    print("═"*60 + "\n")

    try:
        detector = FaceMaskDetector(
            model_path=args.model,
            face_model_path=args.face_model,
            confidence_threshold=args.threshold,
            face_confidence=args.face_conf,
        )
        print("  ✅  Model loaded successfully")
    except Exception as e:
        print(f"  ❌  Model load failed: {e}")
        print("  ℹ️   Run: python src/train.py  to train a model first")
        print("  ℹ️   Or: python scripts/download_dataset.py && python src/train.py")
        sys.exit(1)

    # Alert engine
    alert_engine = AlertEngine(
        sound_enabled=not args.no_alerts,
        log_enabled=args.log,
        screenshot_enabled=args.save_alerts,
        log_path='logs/'
    )

    # Camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"  ❌  Cannot open camera {args.camera}")
        sys.exit(1)

    print(f"  ✅  Camera {args.camera} opened")
    print("\n  Press H in the video window for keyboard controls\n")

    # State
    stats = SessionStats()
    ui = UIOverlay()
    window_name = "FaceMask Detection System v2.0"
    alerts_enabled = not args.no_alerts
    fullscreen = args.fullscreen

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # FPS
    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠️   Frame read failed — retrying...")
                time.sleep(0.05)
                continue

            frame_count += 1

            # FPS calculation
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # ── DETECT ──────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── UPDATE STATS ────────────────────────────────────────
            stats.update(detections)

            # ── ALERTS ──────────────────────────────────────────────
            if alerts_enabled:
                alert_engine.process(frame, detections)

            # ── DRAW ────────────────────────────────────────────────
            ui.draw_detection_boxes(frame, detections, args.show_confidence)
            ui.draw_title_bar(frame, alerts_enabled)
            ui.draw_stats_panel(frame, stats, fps)
            ui.draw_compliance_bar(frame, stats)

            if ui.show_help:
                ui.draw_help(frame)

            # ── DISPLAY ─────────────────────────────────────────────
            cv2.imshow(window_name, frame)

            # ── KEYBOARD ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):          # Q / ESC
                break
            elif key in (ord('s'), ord('S')):             # S — screenshot
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = f"logs/screenshot_{ts}.jpg"
                os.makedirs('logs', exist_ok=True)
                cv2.imwrite(path, frame)
                stats.screenshot_count += 1
                print(f"  📸  Screenshot saved: {path}")
            elif key in (ord('a'), ord('A')):             # A — toggle alerts
                alerts_enabled = not alerts_enabled
                print(f"  🔔  Alerts: {'ON' if alerts_enabled else 'OFF'}")
            elif key in (ord('f'), ord('F')):             # F — fullscreen
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
            elif key in (ord('r'), ord('R')):             # R — reset
                stats.reset()
                print("  🔄  Statistics reset")
            elif key in (ord('h'), ord('H')):             # H — help
                ui.show_help = not ui.show_help

    except KeyboardInterrupt:
        print("\n  ⚡  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_engine.cleanup()

        # Session summary
        print("\n" + "═"*60)
        print("  📊  SESSION SUMMARY")
        print("═"*60)
        print(f"  ⏱   Uptime:        {stats.get_uptime()}")
        print(f"  ✅  With Mask:     {stats.with_mask_count}")
        print(f"  ❌  Without Mask:  {stats.without_mask_count}")
        print(f"  ⚠️   Incorrect:     {stats.incorrect_mask_count}")
        print(f"  📈  Compliance:    {stats.get_compliance_rate():.1f}%")
        print(f"  🚨  Alerts:        {stats.alert_count}")
        print(f"  📸  Screenshots:   {stats.screenshot_count}")
        print("═"*60)
        logger.info(f"Session ended. Compliance: {stats.get_compliance_rate():.1f}%")


if __name__ == '__main__':
    main()
