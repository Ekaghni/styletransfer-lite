"""Style transfer engine. Importing this module loads no models and opens no windows."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageDraw, ImageOps

NUM_STYLES = 26
ARBITRARY_SIZE = 384  # the arbitrary-style transform network is fixed at 384x384
STYLE_SIZE = 256  # ... and the style predictor at 256x256
MIN_SIDE = 32
MODEL_DIR = Path(__file__).parent / "models"

ImageLike = Union[str, "os.PathLike[str]", Image.Image, np.ndarray]
StyleSpec = Union[int, str, Mapping[int, float], Sequence[float]]


class StyleTransferError(Exception):
    """Raised for bad input or an unavailable device."""


def load_image(image: ImageLike) -> Image.Image:
    """Return an RGB PIL image from a path, PIL image or HxWx3 uint8 array."""
    if isinstance(image, Image.Image):
        img = ImageOps.exif_transpose(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        try:
            img = ImageOps.exif_transpose(Image.open(image))
        except FileNotFoundError:
            raise StyleTransferError(f"image not found: {image}") from None
        except OSError as exc:
            raise StyleTransferError(f"cannot read image {image}: {exc}") from None
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def parse_style(spec: StyleSpec) -> np.ndarray:
    """Turn a style spec into a normalised weight vector of length 26.

    Accepts an int (one style), a dict ``{index: weight}``, a list of 26
    weights, or a string such as ``"7"`` or ``"3:0.5,12:0.5"``.
    """
    w = np.zeros(NUM_STYLES, dtype=np.float32)
    if isinstance(spec, str):
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            idx, _, wt = part.partition(":")
            try:
                w[_check_index(int(idx))] += float(wt) if wt else 1.0
            except ValueError:
                raise StyleTransferError(f"bad style spec {spec!r}; use e.g. 7 or 3:0.5,12:0.5") from None
    elif isinstance(spec, (int, np.integer)):
        w[_check_index(int(spec))] = 1.0
    elif isinstance(spec, Mapping):
        for idx, wt in spec.items():
            w[_check_index(int(idx))] += float(wt)
    else:
        arr = np.asarray(spec, dtype=np.float32)
        if arr.shape != (NUM_STYLES,):
            raise StyleTransferError(f"a weight list needs {NUM_STYLES} values, got {arr.shape}")
        w = arr.copy()
    if (w < 0).any() or w.sum() <= 0:
        raise StyleTransferError("style weights must be non-negative and not all zero")
    return w / w.sum()


def _check_index(idx: int) -> int:
    if not 0 <= idx < NUM_STYLES:
        raise StyleTransferError(f"style index must be 0 to {NUM_STYLES - 1}, got {idx}")
    return idx


def _check_strength(strength: float) -> float:
    if not 0.0 <= strength <= 1.0:
        raise StyleTransferError(f"strength must be between 0 and 1, got {strength}")
    return float(strength)


def _fit(img: Image.Image, max_size: Optional[int]) -> Image.Image:
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise StyleTransferError(f"image is too small ({w}x{h}); both sides need at least {MIN_SIDE}px")
    if max_size and max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((max(MIN_SIDE, round(w * scale)), max(MIN_SIDE, round(h * scale))), Image.LANCZOS)
    return img


def _keep_color(stylized: Image.Image, original: Image.Image) -> Image.Image:
    """Keep the stylized brightness but take colour from the original photo."""
    y, _, _ = stylized.convert("YCbCr").split()
    _, cb, cr = original.resize(stylized.size, Image.LANCZOS).convert("YCbCr").split()
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def _mix(stylized: Image.Image, original: Image.Image, strength: float) -> Image.Image:
    if strength >= 1.0:
        return stylized
    return Image.blend(original.resize(stylized.size, Image.LANCZOS), stylized, strength)


def resolve_providers(device: str = "auto") -> list:
    """Map a device name to onnxruntime execution providers."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    table = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
    if device == "auto":
        for name in ("cuda", "dml"):
            if table[name] in available:
                return [table[name], "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in table:
        if table[device] not in available:
            hint = {
                "cuda": 'pip uninstall onnxruntime && pip install onnxruntime-gpu',
                "dml": "pip uninstall onnxruntime && pip install onnxruntime-directml",
            }[device]
            raise StyleTransferError(
                f"device '{device}' is not available in this onnxruntime build "
                f"(found: {', '.join(available)}). Try: {hint}"
            )
        return [table[device], "CPUExecutionProvider"]
    raise StyleTransferError(f"unknown device '{device}'; use auto, cpu, cuda or dml")


class StyleTransfer:
    """Fast neural style transfer with 26 built-in styles or any style image.

    >>> st = StyleTransfer()
    >>> st.stylize("photo.jpg", style=7).save("out.png")
    >>> st.stylize("photo.jpg", style={3: 0.5, 12: 0.5}, strength=0.8)
    >>> st.stylize_with("photo.jpg", "painting.jpg")
    """

    def __init__(self, device: str = "auto", max_size: int = 1024, threads: Optional[int] = None):
        self._requested = device
        self.providers = resolve_providers(device)
        self.max_size = max_size
        self.threads = threads
        self._sessions: Dict[str, "object"] = {}

    @property
    def device(self) -> str:
        first = self.providers[0]
        return {"CUDAExecutionProvider": "cuda", "DmlExecutionProvider": "dml"}.get(first, "cpu")

    def _session(self, name: str):
        if name not in self._sessions:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            if self.threads:
                opts.intra_op_num_threads = self.threads
            path = MODEL_DIR / f"{name}.onnx"
            if not path.exists():
                raise StyleTransferError(f"model file missing: {path}")
            sess = ort.InferenceSession(str(path), opts, providers=self.providers)
            active = sess.get_providers()
            if active and active[0] != self.providers[0]:
                # onnxruntime silently falls back to CPU when CUDA libraries fail to load
                if self._requested != "auto":
                    raise StyleTransferError(
                        f"could not start {self.providers[0]} (onnxruntime fell back to {active[0]}). "
                        "Check your CUDA/cuDNN install, or use --device cpu."
                    )
                self.providers = ["CPUExecutionProvider"]
            self._sessions[name] = sess
        return self._sessions[name]

    # -- built-in 26 styles -------------------------------------------------

    def stylize(
        self,
        image: ImageLike,
        style: StyleSpec = 0,
        strength: float = 1.0,
        preserve_color: bool = False,
        max_size: Optional[int] = None,
    ) -> Image.Image:
        """Apply one of the 26 built-in styles (or a blend of several).

        ``strength`` (0 to 1) mixes the result with the original photo.
        ``max_size`` caps the longer side in pixels (default: the engine's).
        """
        weights = parse_style(style)
        strength = _check_strength(strength)
        original = _fit(load_image(image), max_size if max_size is not None else self.max_size)
        arr = np.asarray(original, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        # the network downsamples by 4, so reflect-pad to a multiple of 4 and crop after
        ph, pw = (-h) % 4, (-w) % 4
        if ph or pw:
            arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        out = self._session("multistyle26").run(None, {"image": x, "style_weights": weights})[0][0]
        out = out.transpose(1, 2, 0)[:h, :w]
        stylized = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        if preserve_color:
            stylized = _keep_color(stylized, original)
        return _mix(stylized, original, strength)

    def contact_sheet(self, image: ImageLike, tile: int = 256, columns: int = 7) -> Image.Image:
        """Render all 26 styles on one image with index labels, to help you pick."""
        original = load_image(image)
        original = _fit(original, None)
        original.thumbnail((tile, tile), Image.LANCZOS)
        tw, th = original.size
        rows = -(-(NUM_STYLES + 1) // columns)
        pad, label = 6, 16
        sheet = Image.new("RGB", (columns * (tw + pad) + pad, rows * (th + label + pad) + pad), "white")
        draw = ImageDraw.Draw(sheet)
        tiles = [("original", original)] + [
            (str(i), self.stylize(original, style=i, max_size=None)) for i in range(NUM_STYLES)
        ]
        for n, (name, im) in enumerate(tiles):
            r, c = divmod(n, columns)
            x, y = pad + c * (tw + pad), pad + r * (th + label + pad)
            sheet.paste(im, (x, y + label))
            draw.text((x + 2, y + 2), name, fill="black")
        return sheet

    # -- arbitrary style from an image ---------------------------------------

    def style_vector(self, style_image: ImageLike) -> np.ndarray:
        """Encode a style image into the 100-number style vector used by stylize_with."""
        img = load_image(style_image)
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((STYLE_SIZE, STYLE_SIZE), Image.LANCZOS)
        x = (np.asarray(img, dtype=np.float32) / 255.0)[None]
        sess = self._session("arbitrary_predict")
        return sess.run(None, {sess.get_inputs()[0].name: x})[0]

    def stylize_with(
        self,
        image: ImageLike,
        style_image: ImageLike,
        strength: float = 1.0,
        preserve_color: bool = False,
        max_size: Optional[int] = None,
    ) -> Image.Image:
        """Use any picture as the style.

        The network works at 384px on the longer side. The result is scaled up
        to the photo's size (capped by ``max_size``) with Lanczos, so it is
        softer than the built-in styles on big photos.
        ``strength`` (0 to 1) interpolates between the photo's own style vector
        and the style image's, so 0 stays close to the photo.
        """
        strength = _check_strength(strength)
        original = _fit(load_image(image), max_size if max_size is not None else self.max_size)
        target = original.size
        vec = self.style_vector(style_image)
        if strength < 1.0:
            own = self.style_vector(original)
            vec = strength * vec + (1.0 - strength) * own

        w, h = target
        scale = ARBITRARY_SIZE / max(w, h)
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = np.asarray(original.resize((sw, sh), Image.LANCZOS), dtype=np.float32) / 255.0
        small = np.pad(small, ((0, ARBITRARY_SIZE - sh), (0, ARBITRARY_SIZE - sw), (0, 0)), mode="reflect")
        sess = self._session("arbitrary_transform")
        names = [i.name for i in sess.get_inputs()]
        out = sess.run(None, {names[0]: small[None], names[1]: vec.astype(np.float32)})[0][0][:sh, :sw]
        stylized = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        if stylized.size != target:
            stylized = stylized.resize(target, Image.LANCZOS)
        if preserve_color:
            stylized = _keep_color(stylized, original)
        return stylized
