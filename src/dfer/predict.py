"""Predict the facial expression in any video file (phone clip, webcam recording, ...).

The face is detected with OpenCV's Haar cascade on 16 uniformly sampled frames, cropped
to a padded square (the last detected box is reused when a frame has no detection),
resized to 224x224 and classified with a fine-tuned model.

Example:
    python -m dfer.predict my_video.mp4 --run-dir runs/videomae_k400_fold1_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dfer.data import EMOTIONS, NUM_CLASSES, ClipTransform
from dfer.models import build_model

_CASCADE = None


def face_detector():
    """OpenCV Haar face detector, loaded on first use (requires opencv-python < 5)."""
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def square_box(x: int, y: int, w: int, h: int, pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Padded square around a face box, clipped to the frame. Returns (x0, y0, x1, y1)."""
    side = int(max(w, h) * (1 + 2 * pad))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    return x0, y0, min(x0 + side, width), min(y0 + side, height)


def face_frames(video_path: str | Path, num_frames: int = 16, pad: float = 0.3) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")
    wanted = set(np.linspace(0, total - 1, num_frames).round().astype(int).tolist())
    frames, box, index = [], None, 0
    while len(frames) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
                box = square_box(x, y, w, h, pad, frame.shape[1], frame.shape[0])
            crop = frame if box is None else frame[box[1]:box[3], box[0]:box[2]]
            frames.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    while len(frames) < num_frames:  # very short videos: repeat the last frame
        frames.append(frames[-1])
    return frames


@torch.no_grad()
def predict(video_path: str | Path, run_dir: str | Path) -> dict[str, float]:
    run_dir = Path(run_dir)
    model_name = json.loads((run_dir / "metrics.json").read_text())["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, NUM_CLASSES)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device, weights_only=True))
    model.to(device).eval()
    clip = ClipTransform(224, train=False)(face_frames(video_path)).unsqueeze(0).to(device)
    probs = torch.softmax(model(clip), dim=1)[0].cpu().numpy()
    return dict(sorted(zip(EMOTIONS, probs.round(4).tolist()), key=lambda kv: -kv[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    for emotion, prob in predict(args.video, args.run_dir).items():
        print(f"{prob:6.1%}  {emotion}")


if __name__ == "__main__":
    main()
