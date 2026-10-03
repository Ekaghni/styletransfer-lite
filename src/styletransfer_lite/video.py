"""Video and webcam stylization. Needs OpenCV: pip install "styletransfer-lite[video]"."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from .core import StyleSpec, StyleTransfer, StyleTransferError


def _cv2():
    try:
        import cv2
    except ImportError:
        raise StyleTransferError(
            'video support needs OpenCV. Install it with: pip install "styletransfer-lite[video]"'
        ) from None
    return cv2


def _frame_to_pil(frame) -> Image.Image:
    return Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB


def _pil_to_frame(img: Image.Image):
    return np.asarray(img)[:, :, ::-1].copy()


def stylize_video(
    src: str,
    dst: str,
    style: StyleSpec = 0,
    engine: Optional[StyleTransfer] = None,
    strength: float = 1.0,
    preserve_color: bool = False,
    max_size: int = 640,
    smooth: float = 0.0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Stylize a video file frame by frame (built-in styles only).

    ``smooth`` (0 to 0.95) blends each output frame with the previous one to
    reduce flicker. There is no true temporal consistency, so some shimmer on
    fine texture is normal. Audio is not copied.
    """
    cv2 = _cv2()
    if not 0.0 <= smooth < 1.0:
        raise StyleTransferError("smooth must be in [0, 1)")
    engine = engine or StyleTransfer()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise StyleTransferError(f"cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    prev = None
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = engine.stylize(
                _frame_to_pil(frame), style=style, strength=strength,
                preserve_color=preserve_color, max_size=max_size,
            )
            arr = np.asarray(out, dtype=np.float32)
            if smooth and prev is not None:
                arr = smooth * prev + (1.0 - smooth) * arr
            prev = arr
            bgr = arr[:, :, ::-1].astype(np.uint8)
            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise StyleTransferError(f"cannot write video: {dst}")
            writer.write(np.ascontiguousarray(bgr))
            n += 1
            if progress:
                progress(n, total)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if n == 0:
        raise StyleTransferError(f"no frames could be read from {src}")
    return {"frames": n, "fps": fps, "output": dst}


def webcam(style: int = 0, camera: int = 0, engine: Optional[StyleTransfer] = None, max_size: int = 480) -> None:
    """Live preview. Keys: n / p = next / previous style, q or Esc = quit."""
    cv2 = _cv2()
    engine = engine or StyleTransfer()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise StyleTransferError(f"cannot open camera {camera}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = engine.stylize(_frame_to_pil(frame[:, ::-1]), style=style, max_size=max_size)
            view = _pil_to_frame(out)
            cv2.putText(view, f"style {style}  (n/p change, q quit)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("styletransfer-lite", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                style = (style + 1) % 26
            elif key == ord("p"):
                style = (style - 1) % 26
    finally:
        cap.release()
        cv2.destroyAllWindows()
