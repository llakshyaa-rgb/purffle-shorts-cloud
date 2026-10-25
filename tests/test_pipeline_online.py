"""The keyed path end to end with fakes: LLM -> Pexels search + download -> render -> upload/schedule/queue."""

import json
import subprocess

import pytest

from purffle_shorts import ffmpeg
from purffle_shorts import media as M
from purffle_shorts import pipeline as P
from purffle_shorts import utils as U

from .conftest import needs_ffmpeg

SCRIPT = {
    "topic": "Octopus hearts", "title": "Why Octopuses Have Three Hearts", "hook_text": "3 HEARTS?!",
    "scenes": [
        {"narration": "An octopus has three hearts.", "search_query": "octopus", "image_prompt": "octopus"},
        {"narration": "Two pump blood through the gills.", "search_query": "octopus gills", "image_prompt": "gills"},
        {"narration": "Its skin changes color to hide among coral.", "search_query": "octopus camouflage", "image_prompt": "an octopus blending into coral"},
    ],
    "description": "Three hearts, one octopus.", "hashtags": ["#octopus"], "tags": ["octopus"],
    "category": "science",
}


class FakeLLM:
    def __init__(self, settings):
        self.providers, self.label = [object()], "fake:model"

    def complete_json(self, system, user, schema=None, max_tokens=4000, temperature=None):
        assert "JSON" in system
        return json.loads(json.dumps(SCRIPT))


class Resp:
    def __init__(self, data=None, path=None):
        self.status_code, self._data, self._path, self.text = 200, data, path, ""

    def json(self):
        return self._data

    def iter_content(self, chunk_size=65536):
        with open(self._path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=6",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def online(settings, monkeypatch, clip):
    searches = []

    class Session:
        def get(self, url, params=None, headers=None, timeout=None, stream=False):
            if "api.pexels.com/videos/search" in url:
                searches.append(params["query"])
                base = abs(hash(params["query"])) % 1000 * 10
                words = " ".join(str(params["query"]).lower().split())
                # Deterministic, descriptive mock metadata: all fixture scenes concern octopuses.
                words = " ".join(dict.fromkeys((words + " octopus octopuses gills blood hearts camouflage coral underwater ocean sea animal nature").split()))
                vids = [{"id": base + i, "duration": 6, "user": {"name": "Ann"},
                         "url": f"https://www.pexels.com/video/{words.replace(' ', '-')}-{base + i}/",
                         "tags": [{"name": word} for word in words.split()],
                         "video_files": [{"link": f"http://fake/{base + i}.mp4", "width": 640, "height": 360,
                                          "file_type": "video/mp4"}]} for i in range(2)]
                return Resp({"videos": vids})
            if url.startswith("http://fake/"):
                return Resp(path=clip)
            raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(M, "http", lambda: Session())
    monkeypatch.setattr(U, "http", lambda: Session())
    monkeypatch.setattr(P, "LLM", FakeLLM)
    s = settings.with_overrides(tts_engine="silent", target_seconds=8, resolution=(360, 640), fps=24, music_volume=0.0,
                                visual_sources=["pexels"], pexels_api_key="k", upload=True)
    return s, searches


@needs_ffmpeg
def test_keyed_pipeline_uploads_and_dedupes(online, monkeypatch):
    s, searches = online
    uploads = []
    monkeypatch.setattr(P.youtube, "upload", lambda settings, video, body: uploads.append(body) or {"id": "vid1"})
    studio = P.Studio(s)
    r = studio.make()
    assert r.ok, r.error
    assert r.status == "uploaded" and r.youtube_id == "vid1"
    assert searches[:1] == ["octopus"]
    body = uploads[0]
    assert body["snippet"]["title"] == "Why Octopuses Have Three Hearts #shorts"
    assert body["snippet"]["categoryId"] == "28"
    assert "Footage: Pexels" in body["snippet"]["description"]
    assert not r.video.exists()                        # KEEP_VIDEOS=false removes the MP4 after upload
    used = studio.history.used_media()
    assert len(used) == 3 and all(k.startswith("pexels:") for k in used)


@needs_ffmpeg
def test_schedule_and_quota_queue(online, monkeypatch):
    s, _ = online
    monkeypatch.setattr(P.youtube, "upload", lambda settings, video, body: {"id": "v"})
    s = s.with_overrides(publish_times=["09:00", "21:00"], keep_videos=True, daily_upload_limit=1)
    studio = P.Studio(s)
    first = studio.make()
    assert first.status == "scheduled" and first.publish_at
    second = studio.make()
    assert second.status == "queued" and second.video.exists()
    assert [row["id"] for row in studio.history.pending_uploads()] == [second.record_id]


@needs_ffmpeg
def test_youtube_refusal_pauses_uploads_until_the_reset(online, monkeypatch):
    s, _ = online
    calls = []

    def refuse(settings, video, body):
        calls.append(video)
        raise P.youtube.QuotaExceeded("uploadLimitExceeded")
    monkeypatch.setattr(P.youtube, "upload", refuse)
    studio = P.Studio(s.with_overrides(keep_videos=True))
    first, second = studio.make(), studio.make()
    assert first.status == second.status == "queued" and len(calls) == 1  # the second one was not even tried
    assert studio.quota_left() == 0                                     # so autopilot sleeps instead of piling up
    assert len(studio.history.pending_uploads()) == 2


@needs_ffmpeg
def test_creator_publish_time_overrides_single_configured_slot(online, monkeypatch):
    from datetime import datetime, timedelta, timezone

    s, _ = online
    uploads = []
    monkeypatch.setattr(P.youtube, "upload", lambda settings, video, body: uploads.append(body) or {"id": "manual"})
    s = s.with_overrides(publish_times=["09:00"], keep_videos=True)
    studio = P.Studio(s)
    made = studio.make(upload=False)
    assert made.ok and made.record_id
    chosen = (datetime.now(timezone.utc) + timedelta(days=2)).replace(microsecond=0)
    chosen_iso = chosen.strftime("%Y-%m-%dT%H:%M:%SZ")
    studio.history.update_video(made.record_id, publish_at=chosen_iso)
    status = studio.upload_record(made.record_id)
    assert status == "scheduled"
    assert uploads[0]["status"]["publishAt"] == chosen_iso


@needs_ffmpeg
def test_creator_publish_time_overrides_configured_slot(online, monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    s, _ = online
    s = s.with_overrides(keep_videos=True, publish_times=["09:00", "21:00"])
    manual_time = (datetime.now(timezone.utc) + timedelta(days=2)).replace(second=0, microsecond=0)
    manual_iso = manual_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(P.youtube, "next_publish_slot",
                        lambda *args, **kwargs: pytest.fail("manual schedule should take priority"))
    uploads = []
    monkeypatch.setattr(P.youtube, "upload",
                        lambda settings, video, body: uploads.append(body) or {"id": "manual-slot"})
    studio = P.Studio(s)
    folder = tmp_path / "manual-schedule"
    folder.mkdir()
    video = folder / "short.mp4"
    video.write_bytes(b"fake video")
    (folder / "metadata.json").write_text(json.dumps({"category_id": "27", "language": "en"}))
    vid = studio.history.add_video(
        title="Manual schedule", description="Test", tags=["test"], status="rendered",
        folder=str(folder), video_path=str(video), duration=10, language="en", publish_at=manual_iso,
    )
    assert studio.upload_record(vid) == "scheduled"
    assert uploads[0]["status"]["publishAt"] == manual_iso


def test_upload_false_is_a_hard_stop_for_direct_upload(settings, monkeypatch, tmp_path):
    s = settings.with_overrides(upload=False, keep_videos=True)
    studio = P.Studio(s)
    folder = tmp_path / "upload-disabled"
    folder.mkdir()
    video = folder / "short.mp4"
    video.write_bytes(b"test-video")
    vid = studio.history.add_video(
        title="Must stay local", status="rendered", folder=str(folder), video_path=str(video),
    )
    monkeypatch.setattr(P.youtube, "upload", lambda *args, **kwargs: pytest.fail("must never upload"))
    assert studio.upload_record(vid) == "disabled"



def test_adaptive_reuse_preserves_intentional_gradient():
    from purffle_shorts.media import MediaItem

    deliberate = MediaItem(
        "generated", "intentional-gradient-2", "generated",
        extra={"intentional_gradient": True},
    )
    missing = MediaItem("generated", "fallback-gradient-2", "generated")
    real = MediaItem("pexels", "video-1", "video")
    assert not P._can_adaptively_reuse(deliberate)
    assert P._can_adaptively_reuse(missing)
    assert not P._can_adaptively_reuse(real)
