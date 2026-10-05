"""Real-time webcam gesture classification demo.

    python demo/webcam_demo.py --checkpoint checkpoints/mobilenet_v2.pt
    python demo/webcam_demo.py --checkpoint checkpoints/cnn.pt --camera 1 --preprocess ir

Place your hand inside the green box. Tips for bridging the near-infrared ->
RGB domain gap: use a plain, dark background and light the hand from the
front; ``--preprocess ir`` darkens the background and boosts the hand so the
input looks more like the Leap Motion training images.

Keys:  q / Esc  quit      p  toggle model-input preview
       s        save the current ROI to data/webcam_snapshots/
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import preprocess_gray  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402


def to_model_gray(roi_bgr: np.ndarray, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "ir":
        # Mimic NIR images: bright hand on near-black background.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    elif mode == "clahe":
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/mobilenet_v2.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--preprocess", choices=["gray", "clahe", "ir"], default="ir")
    ap.add_argument("--roi", type=float, default=0.6, help="ROI box size as fraction of frame height")
    ap.add_argument("--smooth", type=int, default=8, help="average probabilities over N frames")
    ap.add_argument("--threshold", type=float, default=0.5, help="min confidence to show a label")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    classes, size = ckpt["class_names"], ckpt["img_size"]
    print(f"Loaded {ckpt.get('run_name')} ({ckpt['model_name']}), classes={classes}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    hist: deque = deque(maxlen=args.smooth)
    show_input, fps, t_prev = True, 0.0, time.time()
    snap_dir = ROOT / "data" / "webcam_snapshots"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        s = int(h * args.roi)
        x0, y0 = (w - s) // 2, (h - s) // 2
        roi = frame[y0:y0 + s, x0:x0 + s]
        gray = to_model_gray(roi, args.preprocess)

        with torch.no_grad():
            t0 = time.perf_counter()
            probs = model(preprocess_gray(gray, size).to(args.device)).softmax(1)[0].cpu().numpy()
            infer_ms = (time.perf_counter() - t0) * 1000
        hist.append(probs)
        p = np.mean(hist, axis=0)
        top = np.argsort(p)[::-1][:3]

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (0, 200, 0), 2)
        label = classes[top[0]] if p[top[0]] >= args.threshold else "..."
        cv2.putText(frame, f"{label} ({p[top[0]]*100:.0f}%)", (x0, max(30, y0 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        for i, k in enumerate(top):
            y = 30 + 26 * i
            cv2.rectangle(frame, (10, y - 16), (10 + int(180 * p[k]), y + 4), (255, 160, 0), -1)
            cv2.putText(frame, f"{classes[k]} {p[k]*100:.0f}%", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
        cv2.putText(frame, f"{fps:.0f} FPS | model {infer_ms:.1f} ms | {ckpt['model_name']}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if show_input:
            prev = cv2.resize(gray, (160, 160))
            frame[h - 190:h - 30, w - 170:w - 10] = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Hand gesture recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            show_input = not show_input
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            f = snap_dir / f"{int(time.time()*1000)}_{label}.png"
            cv2.imwrite(str(f), gray)
            print("saved", f)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
