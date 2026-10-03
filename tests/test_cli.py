import json
from pathlib import Path

import pytest
from PIL import Image

from styletransfer_lite.cli import main

EX = Path(__file__).parent.parent / "examples"
PHOTO = str(EX / "astronaut.jpg")


def test_default_command_and_json(tmp_path, capsys):
    out = tmp_path / "o.png"
    assert main([PHOTO, "--style", "4", "--max-size", "200", "--json", "--device", "cpu", "-o", str(out)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["style"] == "4" and data["device"] == "cpu" and data["mode"] == "builtin"
    assert Image.open(out).size == tuple(data["size"])


def test_flags_before_positional(tmp_path):
    out = tmp_path / "o.png"
    assert main(["stylize", "--max-size", "128", "-o", str(out), PHOTO]) == 0
    assert out.exists()


def test_blend_and_style_image(tmp_path):
    assert main([PHOTO, "-s", "3:0.5,12:0.5", "--max-size", "128", "-o", str(tmp_path / "a.png")]) == 0
    assert main([PHOTO, "--style-image", str(EX / "great_wave.jpg"), "--max-size", "128",
                 "-o", str(tmp_path / "b.png")]) == 0


def test_folder_batch(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    for n in ("a", "b"):
        Image.new("RGB", (64, 64), (10, 200, 30)).save(folder / f"{n}.png")
    assert main([str(folder), "-o", str(tmp_path / "out"), "-s", "2"]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a_styled.png", "b_styled.png"]


@pytest.mark.parametrize("argv", [[PHOTO, "--style", "99"], ["nope.jpg"], [PHOTO, "--strength", "2"]])
def test_errors_exit_2(argv, capsys):
    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err


def test_sheet_and_info(tmp_path, capsys):
    out = tmp_path / "s.png"
    assert main(["sheet", PHOTO, "-o", str(out), "--tile", "48"]) == 0
    assert out.exists()
    capsys.readouterr()
    assert main(["info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["styles"] == 26


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_video_without_opencv_gives_hint(monkeypatch, capsys):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "cv2":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert main(["video", "x.mp4", "-o", "y.mp4"]) == 2
    assert "styletransfer-lite[video]" in capsys.readouterr().err


def test_gui_import_has_no_side_effects():
    import styletransfer_lite.gui as gui  # must not open a window

    assert callable(gui.launch)
