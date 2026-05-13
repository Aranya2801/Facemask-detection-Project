"""
FaceMask Detection System — Utilities
"""

import os
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Set up a named logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def calculate_compliance_rate(with_mask: int, without_mask: int, incorrect: int) -> float:
    """Calculate compliance rate as percentage."""
    total = with_mask + without_mask + incorrect
    if total == 0:
        return 100.0
    return (with_mask / total) * 100.0


def draw_detections(frame: np.ndarray, detections: List[Dict],
                    show_confidence: bool = True) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    COLORS = {
        'with_mask':      (0, 210, 100),
        'without_mask':   (50,  60, 220),
        'mask_incorrect': (0,  200, 240),
    }
    LABELS = {
        'with_mask':      'MASK ON',
        'without_mask':   'NO MASK',
        'mask_incorrect': 'INCORRECT',
    }

    for det in detections:
        x, y, w, h = det['bbox']
        label      = det.get('label', 'unknown')
        conf       = det.get('confidence', 0.0)
        color      = COLORS.get(label, (200, 200, 200))
        text       = LABELS.get(label, label.upper())
        if show_confidence:
            text += f" {conf:.0%}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(frame, text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 22, 35), 1, cv2.LINE_AA)

    return frame


def resize_with_aspect(frame: np.ndarray, max_width: int = 1280,
                        max_height: int = 720) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def get_frame_stats(detections: List[Dict]) -> Dict:
    """Compute per-frame statistics from detections."""
    counts = {'with_mask': 0, 'without_mask': 0, 'mask_incorrect': 0}
    for det in detections:
        label = det.get('label', '')
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    rate  = calculate_compliance_rate(
        counts['with_mask'], counts['without_mask'], counts['mask_incorrect']
    )
    return {**counts, 'total': total, 'compliance_rate': rate}
