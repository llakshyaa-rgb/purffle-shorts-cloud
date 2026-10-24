from __future__ import annotations

from purffle_shorts.config import Settings


def test_wikimedia_commons_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VISUAL_SOURCES", raising=False)
    assert "commons" in Settings.from_env().visual_sources


def test_visual_sources_environment_override_is_respected(monkeypatch):
    monkeypatch.setenv("VISUAL_SOURCES", "pexels,pixabay")
    assert Settings.from_env().visual_sources == ["pexels", "pixabay"]
