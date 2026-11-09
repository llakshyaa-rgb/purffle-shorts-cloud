import json
from datetime import datetime, timedelta, timezone

from purffle_shorts.analytics import creative_insights
from purffle_shorts.history import History


def _video(history, tmp_path, title, views, hook, ending, retention, style="facts"):
    folder = tmp_path / str(title).replace(" ", "_")
    folder.mkdir()
    (folder / "script.json").write_text(json.dumps({
        "scenes": [
            {"narration": hook, "search_query": "specific opening visual", "image_prompt": "A specific opening shot"},
            {"narration": ending, "search_query": "payoff visual", "image_prompt": "A clear payoff shot"},
        ]
    }), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({
        "visual_coverage_percent": 90,
        "visuals": [{"source": "commons"}, {"source": "pixabay"}],
    }), encoding="utf-8")
    uploaded = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds")
    video_id = history.add_video(
        subject=title, topic=title, source="test", style=style, language="en", title=title,
        description="", tags=[], folder=str(folder), video_path=str(folder / "video.mp4"),
        duration=35, llm="test", voice="test", status="uploaded", youtube_id="yt-" + title,
        uploaded_at=uploaded, score=80,
    )
    history.set_stats("yt-" + title, views, max(1, views // 20), max(0, views // 100))
    history.set_retention("yt-" + title, 20.0, retention, 100.0)
    return video_id


def test_creative_insights_uses_saved_scripts_and_real_metrics(tmp_path):
    history = History(tmp_path / "history.db")
    _video(history, tmp_path, "High performer", 5000, "An ancient device predicted eclipses.", "It worked with turning gears.", 82.0)
    _video(history, tmp_path, "Middle performer", 1200, "A strange discovery changed history.", "The evidence is still there.", 58.0)
    _video(history, tmp_path, "Lower performer", 100, "What if a city vanished?", "Nobody found the city.", 31.0)

    report = creative_insights(history)

    assert "CHANNEL CREATIVE EVIDENCE" in report
    assert "An ancient device predicted eclipses." in report
    assert "It worked with turning gears." in report
    assert "avg viewed 82.0%" in report
    assert "visual coverage 90%" in report
    assert "commons" in report and "pixabay" in report
    assert "does not prove a low swipe-away rate" in report
    assert "only a Studio retention graph can identify" in report


def test_creative_insights_waits_for_three_mature_videos(tmp_path):
    history = History(tmp_path / "history.db")
    _video(history, tmp_path, "One", 100, "Hook one.", "Ending one.", 60.0)
    _video(history, tmp_path, "Two", 200, "Hook two.", "Ending two.", 70.0)
    assert creative_insights(history) == ""


def test_creative_insights_survives_missing_render_artifacts(tmp_path):
    history = History(tmp_path / "history.db")
    for i, views in enumerate((300, 200, 100)):
        uploaded = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds")
        history.add_video(
            subject=f"Topic {i}", topic=f"Topic {i}", source="test", style="facts", language="en",
            title=f"Video {i}", description="", tags=[], folder=str(tmp_path / f"missing-{i}"),
            video_path="", duration=30, llm="test", voice="test", status="uploaded",
            youtube_id=f"missing-{i}", uploaded_at=uploaded,
        )
        history.set_stats(f"missing-{i}", views, 2, 1)
    report = creative_insights(history)
    assert "CHANNEL CREATIVE EVIDENCE" in report
    assert "average-view metrics unavailable" in report


def test_script_doctor_receives_channel_evidence():
    from purffle_shorts.config import Settings
    from purffle_shorts.script import Scene, Script, review_prompt

    script = Script(
        topic="Ancient machines", title="The ancient computer", hook_text="A machine from 2,000 years ago.",
        scenes=[
            Scene("A machine from 2,000 years ago calculated eclipses.", "Antikythera mechanism gears",
                  "Ancient bronze gears inside the Antikythera mechanism, macro museum shot."),
            Scene("Its gears tracked the movements of the sky.", "Antikythera mechanism reconstruction",
                  "Reconstructed bronze astronomical mechanism turning, close-up."),
        ],
        description="", style="facts", language="en",
    )
    evidence = "Higher performer: concrete discovery-first hooks held strong average view percentage."
    _system, prompt = review_prompt(script, Settings(), rewrite=True, channel_insights=evidence)
    assert "CHANNEL PERFORMANCE EVIDENCE" in prompt
    assert evidence in prompt
    assert "do not assume correlation proves causation" in prompt


def test_script_doctor_remains_compatible_without_channel_evidence():
    from purffle_shorts.config import Settings
    from purffle_shorts.script import Scene, Script, review_prompt

    script = Script(
        topic="Space", title="A strange moon", hook_text="This moon rains diamonds.",
        scenes=[
            Scene("This moon may form diamond rain.", "Neptune diamond rain",
                  "Scientific illustration of Neptune interior and carbon crystals."),
        ],
        description="", style="explainer", language="en",
    )
    _system, prompt = review_prompt(script, Settings(), rewrite=True)
    assert "CHANNEL PERFORMANCE EVIDENCE" not in prompt
    assert "Creative-quality requirements" in prompt
