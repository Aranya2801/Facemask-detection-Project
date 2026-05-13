"""
FaceMask Detection System — Alert Engine
=========================================
Handles audio alerts, event logging, screenshots, and webhooks.
"""

import os
import csv
import json
import time
import logging
import threading
from datetime import datetime
from typing import List, Dict
import numpy as np

try:
    import winsound  # Windows
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class AlertEngine:
    """
    Processes detection results and triggers configured alerts.

    Features:
        - Audio beep on non-compliance
        - CSV event logging
        - JSON alert log
        - Screenshot capture on violations
        - Webhook POST (configurable)
        - Cooldown to prevent alert spam
    """

    ALERT_CLASSES = {'without_mask', 'mask_incorrect'}

    def __init__(
        self,
        sound_enabled: bool = True,
        log_enabled: bool   = True,
        screenshot_enabled: bool = False,
        log_path: str = 'logs/',
        cooldown_seconds: float = 3.0,
        webhook_url: str = '',
    ):
        self.sound_enabled     = sound_enabled
        self.log_enabled       = log_enabled
        self.screenshot_enabled = screenshot_enabled
        self.log_path          = log_path
        self.cooldown          = cooldown_seconds
        self.webhook_url       = webhook_url

        self._last_alert_time  = 0.0
        self._alert_count      = 0
        self._lock             = threading.Lock()

        if log_enabled:
            os.makedirs(log_path, exist_ok=True)
            self._init_csv_log()

    def _init_csv_log(self):
        """Initialize CSV log file with headers."""
        self._csv_path = os.path.join(self.log_path, 'detections.csv')
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'label', 'confidence',
                    'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                    'alert_triggered'
                ])

    def process(self, frame: np.ndarray, detections: List[Dict]):
        """Process detections and trigger relevant alerts."""
        if not detections:
            return

        violations = [d for d in detections if d.get('label') in self.ALERT_CLASSES]

        if violations:
            self._trigger_alert(frame, violations)

        if self.log_enabled:
            self._log_detections(detections)

    def _trigger_alert(self, frame: np.ndarray, violations: List[Dict]):
        """Fire alert if cooldown has passed."""
        now = time.time()
        with self._lock:
            if now - self._last_alert_time < self.cooldown:
                return
            self._last_alert_time = now
            self._alert_count += 1

        # Sound alert (non-blocking thread)
        if self.sound_enabled:
            threading.Thread(target=self._play_beep, daemon=True).start()

        # Screenshot
        if self.screenshot_enabled:
            self._save_screenshot(frame, violations)

        # JSON log
        self._log_alert_json(violations)

    def _play_beep(self):
        """Play alert sound."""
        try:
            if WINSOUND_AVAILABLE:
                winsound.Beep(1000, 200)
                time.sleep(0.1)
                winsound.Beep(1000, 200)
            elif PYGAME_AVAILABLE:
                # Generate beep with pygame
                sample_rate = 44100
                duration    = 0.2
                freq        = 880
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                stereo = np.column_stack((wave, wave))
                sound = pygame.sndarray.make_sound(stereo)
                sound.play()
                time.sleep(0.25)
            else:
                # Terminal bell
                print('\a', end='', flush=True)
        except Exception:
            pass

    def _save_screenshot(self, frame: np.ndarray, violations: List[Dict]):
        """Save annotated screenshot of violation."""
        try:
            import cv2
            ts  = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]
            path = os.path.join(self.log_path, 'screenshots', f'alert_{ts}.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame)
        except Exception as e:
            pass

    def _log_detections(self, detections: List[Dict]):
        """Append detections to CSV log."""
        try:
            with open(self._csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                ts = datetime.now().isoformat()
                for det in detections:
                    x, y, w, h = det.get('bbox', (0,0,0,0))
                    is_alert = det.get('label') in self.ALERT_CLASSES
                    writer.writerow([
                        ts, det.get('label',''), f"{det.get('confidence',0):.4f}",
                        x, y, w, h, int(is_alert)
                    ])
        except Exception:
            pass

    def _log_alert_json(self, violations: List[Dict]):
        """Append alert to JSON log."""
        try:
            alert_path = os.path.join(self.log_path, 'alerts.json')
            entry = {
                'timestamp':  datetime.now().isoformat(),
                'violations': [{
                    'label':      v.get('label'),
                    'confidence': round(v.get('confidence', 0), 4),
                } for v in violations]
            }
            alerts = []
            if os.path.exists(alert_path):
                with open(alert_path) as f:
                    try:
                        alerts = json.load(f)
                    except Exception:
                        alerts = []
            alerts.append(entry)
            with open(alert_path, 'w') as f:
                json.dump(alerts[-1000:], f, indent=2)  # Keep last 1000
        except Exception:
            pass

    def cleanup(self):
        """Release resources."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    @property
    def alert_count(self):
        return self._alert_count
