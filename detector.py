"""
FaceMask Detection System — Core Detector Class
================================================
Combines OpenCV DNN face detection + MobileNetV2 classification.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional


class FaceMaskDetector:
    """
    End-to-end face mask detection pipeline.

    Stages:
        1. OpenCV DNN (ResNet-SSD) → detect face regions
        2. MobileNetV2 classifier  → classify each face
        3. Return structured detections

    Example:
        detector = FaceMaskDetector()
        detections = detector.detect(frame)
        for det in detections:
            print(det['label'], det['confidence'], det['bbox'])
    """

    CLASSES = ['with_mask', 'without_mask', 'mask_incorrect']
    IMG_SIZE = (224, 224)

    # BGR colors
    COLORS = {
        'with_mask':      (0, 210, 100),
        'without_mask':   (50,  60, 220),
        'mask_incorrect': (0,  190, 240),
    }

    def __init__(
        self,
        model_path: str = 'models/facemask_model.h5',
        face_model_path: str = 'models/face_detector.caffemodel',
        face_proto_path: str = 'models/face_detector.prototxt',
        confidence_threshold: float = 0.5,
        face_confidence: float = 0.5,
        use_gpu: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.face_confidence = face_confidence
        self.model_path = model_path
        self._face_net = None
        self._mask_model = None

        self._load_face_detector(face_model_path, face_proto_path, use_gpu)
        self._load_mask_classifier(model_path)

    # ── LOADING ─────────────────────────────────────────────────────────────

    def _load_face_detector(self, caffemodel: str, prototxt: str, use_gpu: bool):
        """Load OpenCV DNN face detector (ResNet-SSD)."""
        if not os.path.exists(caffemodel) or not os.path.exists(prototxt):
            print(f"  ⚠️   Face detector files not found.")
            print(f"      Expected: {caffemodel}")
            print(f"      Run: python scripts/download_models.py")
            self._face_net = None
            return

        self._face_net = cv2.dnn.readNet(prototxt, caffemodel)
        if use_gpu:
            self._face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def _load_mask_classifier(self, model_path: str):
        """Load trained MobileNetV2 mask classifier."""
        if not os.path.exists(model_path):
            print(f"  ⚠️   Mask model not found: {model_path}")
            print(f"      Train first: python src/train.py")
            self._mask_model = None
            return

        try:
            # Lazy import TensorFlow (only when model available)
            import tensorflow as tf
            self._mask_model = tf.keras.models.load_model(model_path)
            print(f"  ✅  Mask model loaded: {model_path}")
        except Exception as e:
            print(f"  ❌  Failed to load mask model: {e}")
            self._mask_model = None

    # ── DETECTION ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run full detection pipeline on a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            List of dicts: {bbox, label, confidence, probabilities}
        """
        if frame is None or frame.size == 0:
            return []

        if self._face_net is None or self._mask_model is None:
            # Demo mode — return mock detection for testing UI
            return self._demo_detect(frame)

        faces = self._detect_faces(frame)
        if not faces:
            return []

        detections = []
        for (x, y, w, h) in faces:
            face_roi = self._crop_face(frame, x, y, w, h)
            label, confidence, probs = self._classify_face(face_roi)

            if confidence >= self.confidence_threshold:
                detections.append({
                    'bbox':         (x, y, w, h),
                    'label':        label,
                    'confidence':   confidence,
                    'probabilities': probs,
                })

        return detections

    def detect_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Detect masks in an image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        detections = self.detect(frame)
        return frame, detections

    # ── FACE DETECTION ───────────────────────────────────────────────────────

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """
        Use OpenCV DNN (ResNet-SSD) to find face bounding boxes.

        Returns list of (x, y, w, h) tuples.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.face_confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            fw = x2 - x1
            fh = y2 - y1

            # Filter tiny detections
            if fw < 20 or fh < 20:
                continue

            faces.append((x1, y1, fw, fh))

        return faces

    # ── CLASSIFICATION ───────────────────────────────────────────────────────

    def _crop_face(self, frame: np.ndarray, x: int, y: int,
                   w: int, h: int, padding: float = 0.1) -> np.ndarray:
        """Crop face with optional padding, resize to model input size."""
        fh, fw = frame.shape[:2]

        # Add padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        roi = frame[y1:y2, x1:x2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, self.IMG_SIZE)
        return roi

    def _classify_face(self, face_roi: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Run MobileNetV2 classifier on a cropped face.

        Returns:
            label:       predicted class name
            confidence:  prediction probability (0–1)
            probs:       all class probabilities
        """
        # Preprocess (MobileNetV2 expects [-1, 1])
        x = face_roi.astype('float32')
        x = x / 127.5 - 1.0
        x = np.expand_dims(x, axis=0)

        probs = self._mask_model.predict(x, verbose=0)[0]
        class_idx = np.argmax(probs)
        label = self.CLASSES[class_idx]
        confidence = float(probs[class_idx])

        return label, confidence, probs

    # ── DEMO MODE ────────────────────────────────────────────────────────────

    def _demo_detect(self, frame: np.ndarray) -> List[Dict]:
        """Return mock detections when model is not loaded (for UI testing)."""
        # Use Haar cascade as fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        raw_faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        detections = []
        for (x, y, w, h) in raw_faces:
            # Random label for demo
            label = np.random.choice(self.CLASSES, p=[0.7, 0.2, 0.1])
            conf  = float(np.random.uniform(0.85, 0.99))
            probs = np.array([0.0, 0.0, 0.0])
            probs[self.CLASSES.index(label)] = conf
            detections.append({
                'bbox':          (int(x), int(y), int(w), int(h)),
                'label':         label,
                'confidence':    conf,
                'probabilities': probs,
                'demo_mode':     True,
            })

        return detections

    # ── BATCH ────────────────────────────────────────────────────────────────

    def detect_batch(self, image_paths: List[str]) -> List[Dict]:
        """Detect masks in a batch of images."""
        results = []
        for path in image_paths:
            try:
                frame, detections = self.detect_image(path)
                results.append({'path': path, 'detections': detections, 'error': None})
            except Exception as e:
                results.append({'path': path, 'detections': [], 'error': str(e)})
        return results

    # ── INFO ─────────────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        info = {
            'model_path':    self.model_path,
            'classes':       self.CLASSES,
            'input_size':    self.IMG_SIZE,
            'threshold':     self.confidence_threshold,
            'face_loaded':   self._face_net is not None,
            'mask_loaded':   self._mask_model is not None,
        }
        if self._mask_model is not None:
            info['model_params'] = self._mask_model.count_params()
        return info
