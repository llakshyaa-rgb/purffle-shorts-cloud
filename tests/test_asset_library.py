from pathlib import Path

from PIL import Image

from purffle_shorts import asset_library as AL
from purffle_shorts.asset_library import asset_path, ensure_character_collection, get_or_create_asset, list_assets
from purffle_shorts.visual_library import make_visual


def test_scene_art_is_persisted_and_reused(tmp_path: Path):
    root = tmp_path / ".purffle_assets"
    first = get_or_create_asset(root, "moon orbit", "A clear orbit diagram", 360, 640, variation=2, mode="diagram", style_key="house")
    second = get_or_create_asset(root, "moon orbit", "A clear orbit diagram", 360, 640, variation=2, mode="diagram", style_key="house")
    assert first["id"] == second["id"]
    assert second["uses"] == 2
    assert asset_path(root, first["id"]) == first["path"]
    with Image.open(first["path"]) as image:
        assert image.size == (360, 640)
    assert len(list_assets(root)) == 1


def test_named_character_is_stable_across_scene_prompts(tmp_path: Path):
    root = tmp_path / ".purffle_assets"
    a = get_or_create_asset(root, "first scene", "REUSABLE CHARACTER: purffle_narrator", 360, 640,
                            mode="character", style_key="purffle_narrator")
    b = get_or_create_asset(root, "second scene", "REUSABLE CHARACTER: purffle_narrator", 360, 640,
                            mode="character", style_key="purffle_narrator")
    assert a["id"] == b["id"]
    assert a["character"] == "hooded_narrator"
    assert b["uses"] == 2


def test_character_collection_seeds_canonical_cast_once(tmp_path: Path):
    root = tmp_path / ".purffle_assets"
    cast = ensure_character_collection(root, 360, 640)
    second = ensure_character_collection(root, 360, 640)
    assert len(cast) == 5
    assert {item["id"] for item in cast} == {item["id"] for item in second}
    assert {item["uses"] for item in cast} == {item["uses"] for item in second}
    assert {item["character"] for item in cast} >= {
        "hooded_narrator", "detective", "scientist", "explorer", "mysterious_figure"
    }
    assert len(list_assets(root)) == 5


def test_reusable_cast_has_distinct_visual_identities(tmp_path: Path):
    keys = ["purffle_narrator", "purffle_detective", "purffle_scientist", "purffle_explorer", "purffle_mystery"]
    fingerprints = []
    for key in keys:
        target = tmp_path / f"{key}.png"
        make_visual(key, f"REUSABLE CHARACTER: {key}", target, 360, 640,
                    mode="character", style_key=key)
        with Image.open(target) as image:
            assert image.size == (360, 640)
            fingerprints.append(image.convert("RGB").tobytes())
    assert len(set(fingerprints)) == len(keys)


def test_cache_eviction_preserves_canonical_cast(tmp_path: Path, monkeypatch):
    root = tmp_path / ".purffle_assets"
    monkeypatch.setattr(AL, "MAX_ASSETS", 6)
    cast = ensure_character_collection(root, 360, 640)
    for i in range(4):
        get_or_create_asset(root, f"scene {i}", f"unique illustration {i}", 360, 640, variation=i)
    assert all(asset_path(root, item["id"]) for item in cast)
    assert {item["id"] for item in ensure_character_collection(root, 360, 640)} == {
        item["id"] for item in cast
    }
    assert len(list_assets(root, limit=6)) == 6
