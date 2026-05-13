"""
FaceMask Detection System v2.0 — Flask Web Server
===================================================
Serves the live dashboard, MJPEG stream, and REST API.

Usage:
    python web/app.py
    python web/app.py --port 5000 --camera 0 --host 0.0.0.0
    python web/app.py --api-only   # No webcam, just API endpoints
"""

import os
import sys
import cv2
import json
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from functools import wraps
from flask import (
    Flask, Response, render_template, jsonify,
    request, send_from_directory, abort
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from detector import FaceMaskDetector
from alert_engine import AlertEngine
from utils import calculate_compliance_rate

# ─── APP ──────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder='templates',
            static_folder='.')

app.config['SECRET_KEY'] = os.urandom(24)
app.config['JSON_SORT_KEYS'] = False

# ─── GLOBAL STATE ─────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.session_start  = datetime.now()
        self.with_mask      = 0
        self.without_mask   = 0
        self.mask_incorrect = 0
        self.alerts         = 0
        self.fps            = 0.0
        self.alerts_enabled = True
        self.current_frame  = None
        self.recent_detections = []   # last 10 detection events
        self.lock           = threading.Lock()

    def update(self, detections, fps):
        with self.lock:
            self.fps = fps
            self.recent_detections = []
            for d in detections:
                label = d.get('label', '')
                if label == 'with_mask':
                    self.with_mask += 1
                elif label == 'without_mask':
                    self.without_mask += 1
                    self.alerts += 1
                elif label == 'mask_incorrect':
                    self.mask_incorrect += 1

                self.recent_detections.append({
                    'label':      label,
                    'confidence': round(d.get('confidence', 0), 4),
                    'bbox':       list(d.get('bbox', [])),
                })

    def get_stats(self):
        with self.lock:
            total = self.with_mask + self.without_mask + self.mask_incorrect
            rate  = calculate_compliance_rate(
                self.with_mask, self.without_mask, self.mask_incorrect)
            delta = datetime.now() - self.session_start
            h, r  = divmod(int(delta.total_seconds()), 3600)
            m, s  = divmod(r, 60)
            return {
                'with_mask':      self.with_mask,
                'without_mask':   self.without_mask,
                'mask_incorrect': self.mask_incorrect,
                'alerts':         self.alerts,
                'compliance_rate': round(rate, 2),
                'total_detections': total,
                'fps':            round(self.fps, 1),
                'uptime':         f"{h:02d}:{m:02d}:{s:02d}",
                'alerts_enabled': self.alerts_enabled,
                'recent_detections': self.recent_detections[-10:],
                'session_start':  self.session_start.isoformat(),
            }

state   = AppState()
detector_obj  = None
alert_engine  = None

# ─── CAMERA THREAD ────────────────────────────────────────────────
class CameraThread(threading.Thread):
    """Background thread: read frames, run detection, update state."""

    def __init__(self, camera_idx: int, args):
        super().__init__(daemon=True)
        self.camera_idx = camera_idx
        self.args       = args
        self._stop      = threading.Event()
        self.fps        = 0.0

    def run(self):
        global detector_obj, alert_engine

        cap = cv2.VideoCapture(self.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"  ❌  Cannot open camera {self.camera_idx}")
            return

        frame_count = 0
        t0 = time.time()

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

            # Detect
            detections = []
            if detector_obj:
                detections = detector_obj.detect(frame)

            # Alerts
            if alert_engine and state.alerts_enabled:
                alert_engine.process(frame, detections)

            # Update state
            state.update(detections, self.fps)

            # Draw boxes on frame for stream
            annotated = self._annotate_frame(frame, detections)
            state.current_frame = annotated

        cap.release()

    def _annotate_frame(self, frame, detections):
        """Draw detection boxes + HUD on frame."""
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
            label = det.get('label', '')
            conf  = det.get('confidence', 0)
            color = COLORS.get(label, (200, 200, 200))
            text  = f"{LABELS.get(label, label)} {conf:.0%}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y-th-10), (x+tw+10, y), color, -1)
            cv2.putText(frame, text, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15,22,35), 1, cv2.LINE_AA)

        # FPS overlay
        cv2.putText(frame, f"{self.fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0,210,100), 1, cv2.LINE_AA)
        # Timestamp
        ts = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, ts, (frame.shape[1]-90, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (90,130,160), 1, cv2.LINE_AA)
        return frame

    def stop(self):
        self._stop.set()


# ─── MJPEG GENERATOR ──────────────────────────────────────────────
def generate_frames():
    """Yield MJPEG frames from current_frame."""
    while True:
        frame = state.current_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(1/30)

# ─── ROUTES ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream')
def stream():
    """MJPEG live stream endpoint."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    """System status."""
    return jsonify({
        'status':  'online',
        'version': '2.0.0',
        'model':   'MobileNetV2',
        'camera':  cam_thread.camera_idx if cam_thread else None,
        'uptime':  state.get_stats()['uptime'],
    })

@app.route('/api/stats')
def stats():
    """Real-time session statistics."""
    return jsonify(state.get_stats())

@app.route('/api/detect', methods=['POST'])
def detect_image():
    """
    Detect masks in an uploaded image.
    Accepts: multipart/form-data with 'image' field
    Returns: JSON with detections
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file  = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    if detector_obj is None:
        return jsonify({'error': 'Detector not initialized'}), 503

    detections = detector_obj.detect(frame)
    total      = len(detections)
    counts     = {'with_mask': 0, 'without_mask': 0, 'mask_incorrect': 0}
    for d in detections:
        label = d.get('label', '')
        if label in counts:
            counts[label] += 1

    rate = calculate_compliance_rate(
        counts['with_mask'], counts['without_mask'], counts['mask_incorrect'])

    return jsonify({
        'faces_detected':  total,
        'with_mask':       counts['with_mask'],
        'without_mask':    counts['without_mask'],
        'mask_incorrect':  counts['mask_incorrect'],
        'compliance_rate': round(rate, 2),
        'detections':      detections,
    })

@app.route('/api/screenshot', methods=['POST'])
def screenshot():
    """Save a screenshot of the current frame."""
    frame = state.current_frame
    if frame is None:
        return jsonify({'error': 'No frame available'}), 404

    os.makedirs('logs/screenshots', exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/screenshots/web_{ts}.jpg'
    cv2.imwrite(path, frame)
    return jsonify({'saved': path})

@app.route('/api/logs')
def logs():
    """Fetch event logs."""
    fmt = request.args.get('format', 'json')
    log_path = 'logs/alerts.json'
    if not os.path.exists(log_path):
        return jsonify([])
    with open(log_path) as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/config', methods=['GET', 'PUT'])
def config():
    """Get or update detection config."""
    if request.method == 'GET':
        return jsonify({
            'alerts_enabled':  state.alerts_enabled,
            'threshold':       detector_obj.confidence_threshold if detector_obj else 0.5,
        })
    data = request.get_json(silent=True) or {}
    if 'alerts_enabled' in data:
        state.alerts_enabled = bool(data['alerts_enabled'])
    if 'threshold' in data and detector_obj:
        detector_obj.confidence_threshold = float(data['threshold'])
    return jsonify({'ok': True, 'config': {
        'alerts_enabled': state.alerts_enabled,
    }})

# Static files
@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'css'), filename)

@app.route('/js/<path:filename>')
def js(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'js'), filename)

# ─── MAIN ─────────────────────────────────────────────────────────
cam_thread = None

def parse_args():
    p = argparse.ArgumentParser(description='FaceMask Detection Web Server')
    p.add_argument('--port',     type=int,   default=5000)
    p.add_argument('--host',     type=str,   default='127.0.0.1')
    p.add_argument('--camera',   type=int,   default=0)
    p.add_argument('--model',    type=str,   default='models/facemask_model.h5')
    p.add_argument('--no-alerts',action='store_true')
    p.add_argument('--api-only', action='store_true',
                   help='Run API without camera stream')
    p.add_argument('--debug',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Init detector
    print("\n  🛡️   FaceMask Detection — Web Server v2.0")
    print("  ─" * 30)

    try:
        detector_obj = FaceMaskDetector(model_path=args.model)
    except Exception as e:
        print(f"  ⚠️   Detector init failed: {e} — demo mode")
        detector_obj = None

    # Init alert engine
    alert_engine = AlertEngine(
        sound_enabled    = not args.no_alerts,
        log_enabled      = True,
        screenshot_enabled = True,
        log_path         = 'logs/',
    )

    # Start camera thread
    if not args.api_only:
        cam_thread = CameraThread(args.camera, args)
        cam_thread.start()
        print(f"  ✅  Camera thread started (index {args.camera})")

    print(f"  🌐  Dashboard: http://{args.host}:{args.port}")
    print(f"  📡  API:       http://{args.host}:{args.port}/api/status")
    print(f"  🎬  Stream:    http://{args.host}:{args.port}/api/stream")
    print()

    try:
        app.run(host=args.host, port=args.port,
                debug=args.debug, threaded=True, use_reloader=False)
    finally:
        if cam_thread:
            cam_thread.stop()
        if alert_engine:
            alert_engine.cleanup()
