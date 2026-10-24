from pathlib import Path

from PIL import Image

from purffle_shorts.visual_library import make_visual, visual_type


def test_story_scene_uses_illustration_not_diagram():
    assert visual_type("A boy hears three knocks behind a sealed wardrobe") == "illustration"


def test_story_word_inside_does_not_force_diagram():
    assert visual_type("A tiny room inside the wardrobe glowed in the dark") == "illustration"


def test_explanatory_scene_can_choose_diagram():
    assert visual_type("A diagram explaining how the engine mechanism works") == "diagram"


def test_explicit_process_diagram_is_selected():
    assert visual_type("A step by step diagram of the water cycle") == "diagram"


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


def test_library_creates_diagram_png(tmp_path: Path):
    target = tmp_path / "diagram.png"
    kind = make_visual(
        "A diagram showing how an engine mechanism works",
        "clean infographic",
        target,
        360,
        640,
    )
    assert kind == "diagram"
    with Image.open(target) as image:
        assert image.size == (360, 640)
        assert image.format == "PNG"



def test_specialized_diagrams_render_for_different_subjects(tmp_path: Path):
    cases = [
        ("A diagram of the water cycle", "cycle"),
        ("A timeline of history", "timeline"),
        ("A diagram of the solar system orbit", "orbit"),
        ("A map route between locations", "map"),
        ("A network diagram showing connected nodes", "network"),
    ]
    for index, (query, _) in enumerate(cases):
        target = tmp_path / f"diagram_{index}.png"
        assert make_visual(query, "clear infographic", target, 360, 640) == "diagram"
        with Image.open(target) as image:
            assert image.size == (360, 640)
            assert image.format == "PNG"


def test_more_scene_subjects_create_illustrations(tmp_path: Path):
    cases = [
        "A robot standing in a dark laboratory",
        "A storm above a mountain",
        "A rocket flying through space",
        "A wolf in a snowy forest",
        "A circuit board glowing in the dark",
    ]
    for index, query in enumerate(cases):
        target = tmp_path / f"scene_{index}.png"
        assert make_visual(query, "stylized vector illustration", target, 360, 640) == "illustration"
        with Image.open(target) as image:
            assert image.size == (360, 640)
            assert image.format == "PNG"
