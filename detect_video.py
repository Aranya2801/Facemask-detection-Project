"""
FaceMask Detection System — Video File Detection
==================================================
Usage:
    python detect_video.py --video input.mp4
    python detect_video.py --video input.mp4 --save output.mp4
    python detect_video.py --video input.mp4 --save output.mp4 --report
"""

import cv2
import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detector import FaceMaskDetector
from utils import draw_detections, calculate_compliance_rate


def main():
    parser = argparse.ArgumentParser(description='Detect face masks in a video file')
    parser.add_argument('--video',     required=True, help='Input video path')
    parser.add_argument('--model',     default='models/facemask_model.h5')
    parser.add_argument('--save',      default=None,  help='Output video path')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--skip',      type=int, default=1, help='Process every N frames')
    parser.add_argument('--report',    action='store_true', help='Print per-frame report')
    parser.add_argument('--no-display',action='store_true', help='Skip display window')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"  ❌  Video not found: {args.video}")
        sys.exit(1)

    detector = FaceMaskDetector(
        model_path=args.model,
        confidence_threshold=args.threshold,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"  ❌  Cannot open: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    # Counters
    with_mask = without_mask = incorrect = 0
    frame_idx = 0
    t0 = time.time()

    print(f"\n  🎬  Processing: {os.path.basename(args.video)}")
    print(f"  📏  {width}x{height} @ {fps:.1f}fps | {total} frames\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames
        if frame_idx % args.skip != 0:
            if writer:
                writer.write(frame)
            continue

        detections = detector.detect(frame)
        frame = draw_detections(frame, detections)

        # Stats
        for d in detections:
            label = d.get('label', '')
            if label == 'with_mask':      with_mask += 1
            elif label == 'without_mask': without_mask += 1
            elif label == 'mask_incorrect': incorrect += 1

        # Progress
        elapsed = time.time() - t0
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        pct = frame_idx / total * 100 if total > 0 else 0
        print(f"\r  [{pct:5.1f}%] Frame {frame_idx}/{total} | {proc_fps:.1f} fps", end='')

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow('FaceMask — Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    rate = calculate_compliance_rate(with_mask, without_mask, incorrect)
    print(f"\n\n  📊  RESULTS")
    print("  " + "─"*40)
    print(f"  Frames processed: {frame_idx}")
    print(f"  With mask:        {with_mask}")
    print(f"  Without mask:     {without_mask}")
    print(f"  Incorrect:        {incorrect}")
    print(f"  Compliance rate:  {rate:.1f}%")
    if args.save:
        print(f"  Saved to:         {args.save}")
    print()


if __name__ == '__main__':
    main()
