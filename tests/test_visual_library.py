from pathlib import Path

from PIL import Image

from purffle_shorts.visual_library import make_visual, visual_type


def test_story_scene_uses_illustration_not_diagram():
    assert visual_type("A boy hears three knocks behind a sealed wardrobe") == "illustration"


def test_explanatory_scene_can_choose_diagram():
    assert visual_type("A diagram explaining how the engine mechanism works") == "diagram"


def test_library_creates_vertical_nonphotographic_png(tmp_path: Path):
    target = tmp_path / "scene.png"
    kind = make_visual(
        "A shadow behind a mysterious door",
        "stylized hand-drawn horror illustration",
        target,
        360,
        640,
    )
    assert kind == "illustration"
    assert target.is_file()
    with Image.open(target) as image:
        assert image.size == (360, 640)
        assert image.format == "PNG"
