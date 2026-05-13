"""
FaceMask Detection System — Image Detection
=============================================
Usage:
    python detect_image.py --image photo.jpg
    python detect_image.py --image photo.jpg --save output.jpg --show
"""

import cv2
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detector import FaceMaskDetector
from utils import draw_detections, get_frame_stats


def main():
    parser = argparse.ArgumentParser(description='Detect face masks in an image')
    parser.add_argument('--image',   required=True, help='Input image path')
    parser.add_argument('--model',   default='models/facemask_model.h5')
    parser.add_argument('--save',    default=None, help='Save annotated image to path')
    parser.add_argument('--show',    action='store_true', help='Display result window')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-confidence', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"  ❌  Image not found: {args.image}")
        sys.exit(1)

    detector = FaceMaskDetector(
        model_path=args.model,
        confidence_threshold=args.threshold,
    )

    frame, detections = detector.detect_image(args.image)
    frame = draw_detections(frame, detections, show_confidence=not args.no_confidence)
    stats = get_frame_stats(detections)

    print("\n" + "═"*50)
    print(f"  📸  {os.path.basename(args.image)}")
    print("═"*50)
    print(f"  Faces detected:  {stats['total']}")
    print(f"  With mask:       {stats['with_mask']}")
    print(f"  Without mask:    {stats['without_mask']}")
    print(f"  Incorrect mask:  {stats['mask_incorrect']}")
    print(f"  Compliance rate: {stats['compliance_rate']:.1f}%")
    print("═"*50)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        cv2.imwrite(args.save, frame)
        print(f"  💾  Saved: {args.save}")

    if args.show:
        cv2.imshow('FaceMask Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
