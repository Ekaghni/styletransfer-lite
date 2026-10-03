"""Measure stylization latency. Prints a markdown table.

    python benchmarks/latency.py [--device cpu|cuda|auto] [--runs 10]
"""
import argparse
import platform
import statistics
import time
from pathlib import Path

from PIL import Image

from styletransfer_lite import StyleTransfer

EX = Path(__file__).parent.parent / "examples"


def timed(fn, runs):
    t = time.perf_counter()
    fn()
    cold = time.perf_counter() - t
    warm = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        warm.append(time.perf_counter() - t)
    return cold, statistics.median(warm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    t = time.perf_counter()
    engine = StyleTransfer(device=args.device)
    photo = Image.open(EX / "astronaut.jpg").convert("RGB")
    print(f"python {platform.python_version()}, {platform.platform()}, device: {engine.device}\n")
    print("| case | first call (loads model) | median of warm calls |")
    print("| --- | --- | --- |")
    for side in (256, 512, 800, 1024):
        img = photo.resize((side, side))
        cold, warm = timed(lambda: engine.stylize(img, style=7, max_size=None), args.runs)
        print(f"| built-in style, {side}x{side} | {cold:.2f} s | {warm * 1000:.0f} ms |")
    style = EX / "starry_night.jpg"
    cold, warm = timed(lambda: engine.stylize_with(photo, style), args.runs)
    print(f"| style image, 384px network | {cold:.2f} s | {warm * 1000:.0f} ms |")
    cold, warm = timed(lambda: engine.contact_sheet(photo), 3)
    print(f"| contact sheet (26 styles) | {cold:.2f} s | {warm * 1000:.0f} ms |")


if __name__ == "__main__":
    main()
