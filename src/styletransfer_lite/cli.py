"""Command line interface: ``styletransfer-lite photo.jpg --style 7``."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .core import NUM_STYLES, StyleTransfer, StyleTransferError, parse_style

COMMANDS = ("stylize", "sheet", "video", "webcam", "gui", "info")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"],
                   help="where to run the model (default: auto)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styletransfer-lite",
        description="Fast neural style transfer. 'styletransfer-lite photo.jpg --style 7' works "
                    "without naming a command.",
    )
    parser.add_argument("--version", action="version", version=f"styletransfer-lite {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{stylize,sheet,video,webcam,gui,info}")

    s = sub.add_parser("stylize", help="stylize an image or a folder of images (default command)")
    s.add_argument("input", help="image path, folder, or - for stdin")
    s.add_argument("-o", "--output", help="output file or folder, or - for stdout (PNG)")
    s.add_argument("-s", "--style", default="0",
                   help=f"style index 0-{NUM_STYLES - 1}, or a blend like 3:0.5,12:0.5 (default 0)")
    s.add_argument("--style-image", help="use this picture as the style instead of a built-in one")
    s.add_argument("--strength", type=float, default=1.0, help="0 to 1, how much style to apply (default 1)")
    s.add_argument("--preserve-color", action="store_true", help="keep the photo's original colours")
    s.add_argument("--max-size", type=int, default=1024, help="longer side in pixels (default 1024)")
    _common(s)

    c = sub.add_parser("sheet", help="render all 26 styles on one image to help you choose")
    c.add_argument("input")
    c.add_argument("-o", "--output", default="style_sheet.png")
    c.add_argument("--tile", type=int, default=256, help="tile size in pixels (default 256)")
    _common(c)

    v = sub.add_parser("video", help="stylize a video file (needs the [video] extra)")
    v.add_argument("input")
    v.add_argument("-o", "--output", required=True, help="output .mp4 path")
    v.add_argument("-s", "--style", default="0")
    v.add_argument("--strength", type=float, default=1.0)
    v.add_argument("--preserve-color", action="store_true")
    v.add_argument("--max-size", type=int, default=640, help="longer side in pixels (default 640)")
    v.add_argument("--smooth", type=float, default=0.0, help="0 to 0.95, blend with previous frame to cut flicker")
    _common(v)

    w = sub.add_parser("webcam", help="live webcam preview (needs the [video] extra)")
    w.add_argument("-s", "--style", type=int, default=0)
    w.add_argument("--camera", type=int, default=0)
    w.add_argument("--max-size", type=int, default=480)
    _common(w)

    g = sub.add_parser("gui", help="open the desktop window")
    _common(g)

    i = sub.add_parser("info", help="show version, available devices and model files")
    _common(i)
    return parser


def _emit(args, payload, text: str) -> None:
    print(json.dumps(payload, indent=2) if args.json else text)


def _images_in(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _cmd_stylize(args, out=sys.stdout) -> int:
    engine = StyleTransfer(device=args.device, max_size=args.max_size)
    if not args.style_image:
        parse_style(args.style)  # fail early on a bad spec
    from_stdin = args.input == "-"
    to_stdout = args.output == "-"
    if to_stdout and args.json:
        raise StyleTransferError("--json cannot be combined with -o - (both use stdout)")

    if from_stdin:
        sources = [(io.BytesIO(sys.stdin.buffer.read()), "stdin")]
    else:
        path = Path(args.input)
        if path.is_dir():
            sources = [(p, p.name) for p in _images_in(path)]
            if not sources:
                raise StyleTransferError(f"no images found in {path}")
        elif path.exists():
            sources = [(path, path.name)]
        else:
            raise StyleTransferError(f"not found: {path}")
    batch = len(sources) > 1 or (not from_stdin and Path(args.input).is_dir())

    out_dir = None
    if batch:
        out_dir = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).name + "_stylized")
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for src, name in sources:
        t0 = time.perf_counter()
        kw = dict(strength=args.strength, preserve_color=args.preserve_color, max_size=args.max_size)
        if args.style_image:
            result = engine.stylize_with(src, args.style_image, **kw)
        else:
            result = engine.stylize(src, style=args.style, **kw)
        seconds = time.perf_counter() - t0

        if to_stdout:
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            sys.stdout.buffer.write(buf.getvalue())
            dest = "-"
        else:
            if out_dir is not None:
                dest_path = out_dir / (Path(name).stem + "_styled.png")
            elif args.output:
                dest_path = Path(args.output)
            elif from_stdin:
                dest_path = Path("stdin_styled.png")
            else:
                tag = "custom" if args.style_image else args.style.replace(":", "-").replace(",", "_")
                dest_path = Path(src).with_name(f"{Path(src).stem}_style{tag}.png")
            result.save(dest_path)
            dest = str(dest_path)
        results.append({
            "input": name, "output": dest, "size": list(result.size),
            "style": args.style_image or args.style, "mode": "style-image" if args.style_image else "builtin",
            "strength": args.strength, "device": engine.device, "seconds": round(seconds, 3),
        })

    if to_stdout:
        return 0
    payload = results if batch else results[0]
    lines = [f"{r['input']} -> {r['output']}  ({r['size'][0]}x{r['size'][1]}, {r['seconds']}s on {r['device']})"
             for r in results]
    _emit(args, payload, "\n".join(lines))
    return 0


def _cmd_sheet(args) -> int:
    engine = StyleTransfer(device=args.device)
    t0 = time.perf_counter()
    sheet = engine.contact_sheet(args.input, tile=args.tile)
    sheet.save(args.output)
    payload = {"output": args.output, "size": list(sheet.size), "device": engine.device,
               "seconds": round(time.perf_counter() - t0, 3)}
    _emit(args, payload, f"wrote {args.output} ({sheet.size[0]}x{sheet.size[1]}, {payload['seconds']}s on {engine.device})")
    return 0


def _cmd_video(args) -> int:
    from .video import stylize_video

    engine = StyleTransfer(device=args.device)
    t0 = time.perf_counter()
    last = [0.0]

    def tick(n, total):
        if args.json or time.perf_counter() - last[0] < 1.0:
            return
        last[0] = time.perf_counter()
        print(f"\rframe {n}/{total or '?'}", end="", file=sys.stderr, flush=True)

    info = stylize_video(args.input, args.output, style=args.style, engine=engine, strength=args.strength,
                         preserve_color=args.preserve_color, max_size=args.max_size, smooth=args.smooth,
                         progress=tick)
    secs = time.perf_counter() - t0
    info.update(device=engine.device, seconds=round(secs, 2), fps_processed=round(info["frames"] / secs, 2))
    if not args.json:
        print(file=sys.stderr)
    _emit(args, info, f"wrote {info['output']}: {info['frames']} frames in {secs:.1f}s "
                      f"({info['fps_processed']} fps on {engine.device})")
    return 0


def _cmd_webcam(args) -> int:
    from .video import webcam

    webcam(style=args.style, camera=args.camera, engine=StyleTransfer(device=args.device), max_size=args.max_size)
    return 0


def _cmd_gui(args) -> int:
    from .gui import launch

    return launch(device=args.device)


def _cmd_info(args) -> int:
    import onnxruntime as ort

    from .core import MODEL_DIR, resolve_providers

    payload = {
        "version": __version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "auto_device": resolve_providers("auto")[0],
        "models": sorted(p.name for p in MODEL_DIR.glob("*.onnx")),
        "styles": NUM_STYLES,
    }
    text = "\n".join(f"{k}: {v}" for k, v in payload.items())
    _emit(args, payload, text)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "stylize")
    elif argv and argv[0] == "-":  # bare stdin
        argv.insert(0, "stylize")
    parser = _build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler = {
        "stylize": _cmd_stylize, "sheet": _cmd_sheet, "video": _cmd_video,
        "webcam": _cmd_webcam, "gui": _cmd_gui, "info": _cmd_info,
    }[args.command]
    try:
        return handler(args)
    except StyleTransferError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
