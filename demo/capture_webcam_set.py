"""Record the small self-collected webcam set used for the domain-shift test
(synopsis: ~50-100 images per class).

    python demo/capture_webcam_set.py --checkpoint checkpoints/cnn.pt   # reads class names from it
    python demo/capture_webcam_set.py                                  # default LeapGestRecog names

Keys: 0-9 select the class; SPACE toggles burst-saving (5 img/s) for the
selected class; q quits. Images are saved (after the same preprocessing as
the live demo) to data/webcam/<class>/. Then evaluate with:

    python -m src.evaluate --checkpoint checkpoints/cnn.pt --webcam-dir data/webcam
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.webcam_demo import to_model_gray  # noqa: E402

DEFAULT_CLASSES = ["palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c", "down"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "webcam"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--preprocess", choices=["gray", "clahe", "ir"], default="ir")
    ap.add_argument("--roi", type=float, default=0.6)
    args = ap.parse_args()
    classes = DEFAULT_CLASSES
    if args.checkpoint:
        classes = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["class_names"]

    cap = cv2.VideoCapture(args.camera)
    cls, burst, last = 0, False, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        s = int(h * args.roi)
        x0, y0 = (w - s) // 2, (h - s) // 2
        gray = to_model_gray(frame[y0:y0 + s, x0:x0 + s], args.preprocess)
        d = Path(args.out) / classes[cls]
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        if burst and time.time() - last > 0.2:
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{int(time.time()*1000)}.png"), gray)
            last = time.time()
        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (0, 0, 255) if burst else (0, 200, 0), 2)
        cv2.putText(frame, f"[{cls}] {classes[cls]}  saved: {n}  {'REC' if burst else ''}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "0-9 class | SPACE record | q quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cv2.imshow("capture", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            burst = not burst
        if ord("0") <= k <= ord("9") and k - ord("0") < len(classes):
            cls, burst = k - ord("0"), False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
