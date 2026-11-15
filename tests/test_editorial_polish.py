from purffle_shorts.pipeline import trimmed_voice_duration


def test_trimmed_voice_duration_removes_long_trailing_silence():
    assert trimmed_voice_duration(26.0, [(4.0, 4.4), (20.0, 26.0)]) == 20.1


def test_trimmed_voice_duration_keeps_short_natural_breath():
    assert trimmed_voice_duration(20.0, [(19.7, 20.0)]) == 20.0


def test_trimmed_voice_duration_ignores_internal_silence():
    assert trimmed_voice_duration(20.0, [(8.0, 9.0)]) == 20.0


def test_trimmed_voice_duration_handles_missing_audio_pauses():
    assert trimmed_voice_duration(20.0, []) == 20.0

from types import SimpleNamespace

from PIL import Image

from purffle_shorts.overlays import number_badge_png


def test_number_badge_is_a_transparent_high_contrast_overlay(tmp_path):
    settings = SimpleNamespace(unit=1080, width=1080, height=1920)
    path, x, y = number_badge_png(settings, 2, tmp_path / "badge.png")
    with Image.open(path) as badge:
        assert badge.mode == "RGBA"
        assert badge.width > 100 and badge.height > 100
        assert badge.getchannel("A").getbbox() is not None
    assert x > 0 and 0 < y < settings.height // 2
