from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from styletransfer_lite import NUM_STYLES, StyleTransfer, StyleTransferError, parse_style
from styletransfer_lite.core import resolve_providers

EX = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def engine():
    return StyleTransfer(device="cpu")


@pytest.fixture
def photo():
    rng = np.random.RandomState(1)
    base = rng.rand(10, 14, 3)
    img = Image.fromarray((base * 255).astype(np.uint8)).resize((90, 70), Image.BICUBIC)
    return img


def test_parse_style_variants():
    assert parse_style(3)[3] == 1.0
    w = parse_style("3:1,12:3")
    assert w[3] == pytest.approx(0.25) and w[12] == pytest.approx(0.75)
    assert parse_style({0: 1, 1: 1})[:2].tolist() == [0.5, 0.5]
    assert parse_style([1] * NUM_STYLES).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [26, -1, "x", "3:abc", {99: 1}, [1, 2, 3], "0:0"])
def test_parse_style_rejects_bad(bad):
    with pytest.raises(StyleTransferError):
        parse_style(bad)


def test_output_size_matches_input_for_odd_sizes(engine):
    for size in [(67, 45), (33, 101), (64, 64)]:
        img = Image.new("RGB", size, (120, 80, 40))
        assert engine.stylize(img, style=2).size == size


def test_max_size_limits_long_side(engine, photo):
    big = photo.resize((300, 200))
    out = engine.stylize(big, style=1, max_size=100)
    assert max(out.size) == 100


def test_too_small_image(engine):
    with pytest.raises(StyleTransferError):
        engine.stylize(Image.new("RGB", (10, 10)), style=0)


def test_strength_zero_returns_original(engine, photo):
    out = engine.stylize(photo, style=5, strength=0.0)
    assert np.abs(np.asarray(out, float) - np.asarray(photo, float)).max() <= 1


def test_strength_out_of_range(engine, photo):
    with pytest.raises(StyleTransferError):
        engine.stylize(photo, strength=1.5)


def test_styles_differ(engine, photo):
    a = np.asarray(engine.stylize(photo, style=3), float)
    b = np.asarray(engine.stylize(photo, style=17), float)
    assert np.abs(a - b).mean() > 5


def test_preserve_color_keeps_chroma(engine):
    img = Image.open(EX / "astronaut.jpg").resize((128, 128))
    out = engine.stylize(img, style=3, preserve_color=True)
    cb_o = np.asarray(img.convert("YCbCr"), float)[..., 1]
    cb_s = np.asarray(out.convert("YCbCr"), float)[..., 1]
    assert np.abs(cb_o - cb_s).mean() < 3


def test_rgba_and_array_inputs(engine):
    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
    assert engine.stylize(rgba).mode == "RGB"
    assert engine.stylize(np.zeros((64, 64, 3), np.uint8)).size == (64, 64)


def test_missing_file(engine):
    with pytest.raises(StyleTransferError):
        engine.stylize("does-not-exist.jpg")


def test_style_image_mode(engine):
    out = engine.stylize_with(EX / "astronaut.jpg", EX / "starry_night.jpg", max_size=300)
    assert out.size[0] <= 300 and out.mode == "RGB"
    mean_rgb = np.asarray(out, float).reshape(-1, 3).mean(0)
    assert mean_rgb[2] > mean_rgb[0]  # Starry Night pushes the result toward blue


def test_style_image_extreme_aspect(engine):
    wide = Image.new("RGB", (500, 40), (90, 90, 90))
    assert engine.stylize_with(wide, EX / "great_wave.jpg").size == (500, 40)


def test_style_vector_shape(engine):
    assert engine.style_vector(EX / "great_wave.jpg").shape == (1, 1, 1, 100)


def test_contact_sheet(engine):
    sheet = engine.contact_sheet(EX / "astronaut.jpg", tile=64, columns=7)
    assert sheet.width > 7 * 64 and sheet.height > 4 * 64


def test_unknown_device():
    with pytest.raises(StyleTransferError):
        resolve_providers("tpu")


def test_cuda_error_is_friendly_when_missing():
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA is available here")
    with pytest.raises(StyleTransferError, match="onnxruntime-gpu"):
        resolve_providers("cuda")
