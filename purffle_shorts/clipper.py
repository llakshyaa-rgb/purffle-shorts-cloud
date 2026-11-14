"""Clip a long video (podcast, interview, lecture, stream) into Shorts:

    purffle-shorts clip talk.mp4 --count 3
    purffle-shorts clip "https://..." --crop blur        (URLs need: pip install yt-dlp)

The audio is transcribed with word timings (faster-whisper locally, or OpenAI's API), the LLM reads the
timestamped transcript and picks the moments that work on their own — a hook in the first seconds and a
complete thought at the end — and each one is cropped to 9:16, captioned word by word and rendered.

Only clip videos you own or have permission to reuse.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median

from . import ffmpeg, tts
from .config import Settings
from .llm import LLMConfigError
from .overlays import build_overlays
from .render import _encode_args, _grade, compose, pick_cover
from .script import normalize
from .timing import Word, caption_chunks, srt
from .utils import PermanentError, download, slugify

log = logging.getLogger("purffle")

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook_text": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "title", "hook_text", "description", "hashtags", "tags", "score",
                             "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["clips"],
    "additionalProperties": False,
}
CROPS = ("center", "blur", "face")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
MAX_TRANSCRIPT_CHARS = 60_000


@dataclass
class Sentence:
    start: float
    end: float
    text: str


@dataclass
class ClipPlan:
    start: float
    end: float
    title: str
    hook_text: str = ""
    description: str = ""
    hashtags: list[str] | None = None
    tags: list[str] | None = None
    score: int | None = None     # the LLM's retention estimate; None when no LLM chose the clip
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------------------------------------------------------------------------ source & transcript
def fetch_source(src: str, dest_dir: Path) -> Path:
    if re.match(r"^https?://", src):
        try:
            import yt_dlp
        except ImportError as e:
            raise PermanentError("Clipping from a URL needs yt-dlp:  pip install yt-dlp  "
                                 "(or download the video yourself and pass the file)") from e
        dest_dir.mkdir(parents=True, exist_ok=True)
        opts = {"outtmpl": str(dest_dir / "source.%(ext)s"), "quiet": True, "noprogress": True,
                "format": "bv*+ba/b", "merge_output_format": "mp4"}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([src])
        found = sorted(dest_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
        if not found:
            raise RuntimeError(f"yt-dlp downloaded nothing from {src}")
        return found[0]
    p = Path(src).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"video not found: {p}")
    return p


def transcribe(video: Path, work: Path, settings: Settings) -> list[Word]:
    work.mkdir(parents=True, exist_ok=True)
    audio = work / "audio.mp3"
    ffmpeg.run(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(audio)], label="extract audio")
    lang = settings.language.split("-")[0]
    try:
        model = tts.whisper_model(settings)
        words = tts._align_faster_whisper(audio, lang, model, vad=True)
        log.info("Transcribed %d words with faster-whisper (%s)", len(words), model)
        return words
    except ImportError:
        pass
    if settings.openai_api_key:
        if audio.stat().st_size > 24 * 1024 * 1024:
            raise PermanentError("That video is too long for OpenAI transcription (25 MB audio limit, about "
                                 "70 minutes). Install faster-whisper to clip it locally: pip install faster-whisper")
        words = tts._align_openai(audio, lang, settings.openai_api_key)
        log.info("Transcribed %d words with OpenAI", len(words))
        return words
    raise PermanentError("Clipping needs a transcript: pip install faster-whisper (free, local) "
                         "or set OPENAI_API_KEY")


def sentences(words: list[Word], max_words: int = 30) -> list[Sentence]:
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        pause = (nxt.start - w.end) if nxt else 9.0
        if w.text.rstrip()[-1:] in ".!?" or pause > 0.9 or len(cur) >= max_words:
            out.append(Sentence(cur[0].start, cur[-1].end, " ".join(x.text.strip() for x in cur)))
            cur = []
    return out


def transcript_block(sents: list[Sentence]) -> str:
    lines, size = [], 0
    for s in sents:
        line = f"[{s.start:.1f}-{s.end:.1f}] {s.text}"
        size += len(line) + 1
        if size > MAX_TRANSCRIPT_CHARS:
            lines.append("[...transcript truncated...]")
            break
        lines.append(line)
    return "\n".join(lines)


def sample_visual_frames(source: Path, duration: float, work: Path, limit: int = 12) -> list[dict]:
    """Extract a small, evenly spaced visual storyboard for multimodal moment analysis."""
    if duration <= 0:
        return []
    frames = work / "visual-samples"
    frames.mkdir(parents=True, exist_ok=True)
    interval = max(1.0, duration / max(1, limit))
    ffmpeg.run(["-i", str(source), "-vf", f"fps={1.0 / interval:.6f},scale=384:-2:flags=fast_bilinear",
                "-frames:v", str(limit), "-q:v", "7", str(frames / "frame-%02d.jpg")],
               label="sample visual storyboard")
    out = []
    for i, path in enumerate(sorted(frames.glob("frame-*.jpg"))):
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        out.append({"timestamp": round(i * interval, 2), "mime": "image/jpeg", "data": data})
    return out


def native_visual_features(source: Path, duration: float, work: Path, *, sample_every: float = 1.5,
                           max_samples: int = 240) -> list[dict]:
    """Extract lightweight, model-free visual signals with FFmpeg + Pillow.

    Measures scene changes, frame-to-frame change, luminance, colour spread and edge density.
    This is genuine pixel analysis, not semantic object recognition; it works without an LLM/API.
    """
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError:
        log.warning("Native visual analysis needs Pillow; install the core Purffle dependencies")
        return []
    if duration <= 0:
        return []
    every = max(sample_every, duration / max_samples)
    folder = work / "native-vision"
    folder.mkdir(parents=True, exist_ok=True)
    pattern = folder / "frame-%04d.jpg"
    ffmpeg.run(["-i", str(source), "-vf",
                f"fps={1.0 / every:.6f},scale=96:54:flags=area,format=rgb24",
                "-frames:v", str(max_samples), "-q:v", "8", str(pattern)],
               label="native visual feature scan")
    frames = []
    previous = None
    for index, path in enumerate(sorted(folder.glob("frame-*.jpg"))):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                gray = image.convert("L")
                stat = ImageStat.Stat(gray)
                luminance = float(stat.mean[0]) / 255.0
                contrast = min(1.0, float(stat.stddev[0]) / 96.0)
                # Edge density from neighbouring pixel differences; no neural model required.
                pixels = list(gray.getdata())
                width, height = gray.size
                horizontal = sum(abs(pixels[y * width + x] - pixels[y * width + x + 1])
                                 for y in range(height) for x in range(width - 1))
                vertical = sum(abs(pixels[y * width + x] - pixels[(y + 1) * width + x])
                               for y in range(height - 1) for x in range(width))
                edge = min(1.0, (horizontal + vertical) / (2 * width * height * 255 * 0.22))
                hist = image.histogram()
                total = max(1, sum(hist))
                colour_spread = min(1.0, sum(1 for value in hist if value > total * 0.002) / 96.0)
                change = 0.0
                if previous is not None:
                    diff = ImageChops.difference(gray, previous)
                    change = min(1.0, float(ImageStat.Stat(diff).mean[0]) / 64.0)
                frames.append({"time": min(duration, index * every), "luminance": luminance,
                               "contrast": contrast, "edge": edge, "colour": colour_spread,
                               "change": change})
                previous = gray.copy()
        except Exception as exc:
            log.debug("Skipping native vision frame %s: %s", path, exc)
    return frames


def rank_native_visual_moments(features: list[dict], duration: float, count: int,
                               min_seconds: int, max_seconds: int) -> list[ClipPlan]:
    """Rank non-overlapping moments using measurable visual activity and scene diversity."""
    if len(features) < 2 or duration <= 0:
        return []
    span = min(float(max_seconds), duration)
    if span < min(1, min_seconds):
        return []
    span = max(min(float(min_seconds), duration), min(span, max(float(min_seconds), 32.0)))
    step = max(5.0, span * 0.45)
    starts = []
    cursor = 0.0
    while cursor + span <= duration + 0.01 and len(starts) < 240:
        starts.append(cursor)
        cursor += step
    if not starts:
        starts = [max(0.0, (duration - span) / 2)]
    plans = []
    for start in starts:
        end = min(duration, start + span)
        inside = [f for f in features if start <= f["time"] <= end]
        if len(inside) < 2:
            continue
        changes = [f["change"] for f in inside[1:]]
        # Reward scene transitions, movement, detail and varied composition; avoid blank/static stretches.
        scene_events = sum(1 for value in changes if value >= 0.34)
        motion = sum(changes) / max(1, len(changes))
        visual_detail = sum(f["edge"] for f in inside) / len(inside)
        contrast = sum(f["contrast"] for f in inside) / len(inside)
        colour = sum(f["colour"] for f in inside) / len(inside)
        score = int(max(0, min(100, 100 * (
            0.34 * min(1.0, scene_events / 3.0) +
            0.28 * min(1.0, motion * 2.2) +
            0.18 * visual_detail + 0.12 * contrast + 0.08 * colour))))
        if score < 1:
            continue
        plans.append(ClipPlan(start, end, f"Visual activity moment ({_ts(start)}–{_ts(end)})",
                              reason=f"Native pixel analysis: {scene_events} strong visual changes; "
                                     f"motion {motion:.2f}, detail {visual_detail:.2f}",
                              score=score))
    plans.sort(key=lambda plan: plan.score or 0, reverse=True)
    selected = []
    for plan in plans:
        if any(plan.start < chosen.end and plan.end > chosen.start for chosen in selected):
            continue
        selected.append(plan)
        if len(selected) >= count:
            break
    return selected



def enrich_native_visual_context(source: Path, duration: float, work: Path,
                                 features: list[dict], *, every: float = 2.0,
                                 max_samples: int = 180) -> dict:
    """Optional local CV/OCR pass: face/eye presence, rough track continuity and on-screen text.

    OpenCV and Tesseract are opportunistic local tools. If absent, the core pixel analyser still works.
    Face tracks are anonymous shot-local tracks, never biometric identity recognition.
    """
    result = {"face_frames": 0, "eye_frames": 0, "face_tracks": 0, "ocr_frames": 0,
              "texts": [], "animation_like_frames": 0, "scene_changes": 0}
    if not features or duration <= 0:
        return result
    # Mark hard visual transitions from the already-computed frame differences.
    for frame in features:
        frame["scene_cut"] = bool(frame.get("change", 0.0) >= 0.42)
    result["scene_changes"] = sum(1 for frame in features if frame.get("scene_cut"))
    try:
        import cv2
    except ImportError:
        cv2 = None
    folder = work / "native-context"
    folder.mkdir(parents=True, exist_ok=True)
    interval = max(every, duration / max_samples)
    pattern = folder / "context-%04d.jpg"
    try:
        ffmpeg.run(["-i", str(source), "-vf",
                    f"fps={1.0 / interval:.6f},scale=320:180:flags=area",
                    "-frames:v", str(max_samples), "-q:v", "7", str(pattern)],
                   label="native face and text scan")
    except Exception as exc:
        log.warning("Native context frame extraction failed: %s", str(exc)[:160])
        return result
    face_cascade = eye_cascade = None
    if cv2 is not None and hasattr(cv2, "CascadeClassifier"):
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
            if face_cascade.empty():
                face_cascade = None
            if eye_cascade.empty():
                eye_cascade = None
        except Exception:
            face_cascade = eye_cascade = None
    tracks = []
    for index, path in enumerate(sorted(folder.glob("context-*.jpg"))):
        timestamp = min(duration, index * interval)
        frame = next((item for item in features if abs(item["time"] - timestamp) <= interval * 0.8), None)
        image = None
        if cv2 is not None and face_cascade is not None:
            image = cv2.imread(str(path))
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4,
                                                      minSize=(22, 22))
                if len(faces):
                    result["face_frames"] += 1
                    result["face_frames"] = min(result["face_frames"], max_samples)
                    for (x, y, width, height) in faces[:5]:
                        centre = ((x + width / 2) / max(1, image.shape[1]),
                                  (y + height / 2) / max(1, image.shape[0]))
                        match = next((track for track in tracks
                                      if timestamp - track["last"] <= 5.0 and
                                      ((track["centre"][0] - centre[0]) ** 2 +
                                       (track["centre"][1] - centre[1]) ** 2) ** 0.5 < 0.22), None)
                        if match is None:
                            tracks.append({"first": timestamp, "last": timestamp, "centre": centre, "hits": 1})
                        else:
                            match["last"] = timestamp
                            match["centre"] = centre
                            match["hits"] += 1
                        if eye_cascade is not None:
                            roi = gray[y:y + height, x:x + width]
                            eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3,
                                                                minSize=(5, 5))
                            if len(eyes) >= 1:
                                result["eye_frames"] += 1
                # Animation-like is a visual texture heuristic, not a content classifier.
                if frame and frame.get("edge", 0) > 0.42 and frame.get("colour", 0) > 0.42:
                    result["animation_like_frames"] += 1
        # OCR is optional and stays on-device. Bound calls to keep mobile processing practical.
        if shutil.which("tesseract") and index % 3 == 0:
            try:
                proc = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11"],
                                      capture_output=True, text=True, timeout=5, check=False)
                detected = " ".join(proc.stdout.split())
                if len(detected) >= 4 and detected not in result["texts"]:
                    result["texts"].append(detected[:160])
                    result["ocr_frames"] += 1
                    if len(result["texts"]) >= 20:
                        break
            except (OSError, subprocess.TimeoutExpired):
                pass
    result["face_tracks"] = sum(1 for track in tracks if track["hits"] >= 2)
    result["face_track_spans"] = [
        {"start": round(track["first"], 2), "end": round(track["last"], 2), "observations": track["hits"]}
        for track in tracks if track["hits"] >= 2
    ][:30]
    result["texts"] = result["texts"][:20]
    return result


def apply_visual_context_scores(features: list[dict], context: dict) -> None:
    """Fold local face/eye, scene-cut and animation-texture signals into moment-ranking features."""
    if not features:
        return
    face_density = min(1.0, context.get("face_frames", 0) / max(1, len(features)))
    eye_density = min(1.0, context.get("eye_frames", 0) / max(1, len(features)))
    animation_density = min(1.0, context.get("animation_like_frames", 0) / max(1, len(features)))
    for frame in features:
        frame["face_signal"] = face_density
        frame["eye_signal"] = eye_density
        frame["animation_texture"] = animation_density

def pick_visual_clips(llm, images: list[dict], duration: float, sents: list[Sentence],
                      settings: Settings, count: int, min_s: int, max_s: int,
                      brief: str = "") -> list[ClipPlan]:
    """Use sampled frames plus any transcript to rank moments; falls back cleanly if the model lacks vision."""
    if not images or not callable(getattr(llm, "complete_vision_json", None)):
        return []
    transcript = transcript_block(sents) if sents else "(No transcript was available; use visual evidence only.)"
    ask = f"Creator brief: {brief.strip()}\n" if brief.strip() else ""
    system = ("You are an expert short-form video editor and visual story analyst. Return one JSON object only. "
              "Do not invent events not visible in the supplied frames or stated in the transcript.")
    user = f"""Analyze this video storyboard. Each attached frame is labeled with its timestamp in seconds.
{ask}Transcript with timestamps (when available):
{transcript}

Find up to {count} distinct candidate moments for short vertical edits.
Each moment must be {min_s}-{max_s} seconds, fit inside the full video duration ({duration:.1f}s),
avoid overlap, and have a clear hook, visual change/action or meaningful spoken beat, and payoff.
Use the frame timestamps to ground the start/end estimates. If a transcript exists, align boundaries to
complete thoughts and use the visuals to judge context. If no transcript exists, choose moments from visible
scene changes/actions only; do not pretend speech was analyzed.
Return title, hook_text, description, hashtags, tags, score 0-100 and a concise reason for every candidate.
Return JSON matching the supplied schema."""
    data = llm.complete_vision_json(system, user, images, CLIP_SCHEMA, max_tokens=6000, temperature=0.2)
    plans: list[ClipPlan] = []
    for item in data.get("clips") or []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if sents:
            snapped = _snap(sents, start, end, min_s, max_s)
            if not snapped:
                continue
            start, end = snapped
        else:
            start, end = max(0.0, start), min(duration, end)
            if end - start < min_s or end - start > max_s:
                continue
        if start < 0 or end > duration or end <= start:
            continue
        if any(start < p.end and end > p.start for p in plans):
            continue
        plans.append(ClipPlan(start, end, str(item.get("title") or "Visual moment").strip(),
                              str(item.get("hook_text") or "").strip(),
                              str(item.get("description") or "").strip(),
                              list(item.get("hashtags") or []), list(item.get("tags") or []),
                              max(0, min(100, int(item.get("score") or 0))),
                              str(item.get("reason") or "Visual and transcript analysis")))
    plans.sort(key=lambda p: p.score or 0, reverse=True)
    return plans[:count]


# ------------------------------------------------------------------------------------------ choosing moments
def clip_prompt(sents: list[Sentence], count: int, min_s: int, max_s: int, settings: Settings,
                brief: str = "") -> tuple[str, str]:
    system = ("You are a short-form video editor who turns long videos into viral YouTube Shorts. You reply with a "
              "single JSON object and nothing else.")
    ask = (f"\nThe creator's brief, which comes first: {brief.strip()}\nOnly pick moments that fit this brief.\n"
           if brief.strip() else "")
    user = f"""Below is the transcript of a long video, one sentence per line with [start-end] times in seconds.
{ask}
Pick the {count} best moments to publish as standalone YouTube Shorts:
- each {min_s}-{max_s} seconds long, starting at a sentence start and ending at a sentence end;
- the first sentence must hook a viewer who has NOT seen the rest (a bold claim, a question, a surprising fact);
- the moment must make complete sense on its own and end on a payoff or complete thought;
- no overlapping moments; prefer emotional, funny, surprising or highly useful moments.

For each: start, end (seconds, copied from the transcript), title (max 60 characters, honest, curiosity-driven),
hook_text (max 6 words shown on screen at the start), description (2 sentences + a question), 3-5 hashtags,
8-12 tags, score (0-100: how well it will hold viewers), reason (one sentence). Language of the video: {settings.language}.

Transcript:
{transcript_block(sents)}

Return JSON: {{"clips": [...]}}"""
    return system, user


def _title_from(text: str) -> str:
    """A title from a spoken sentence (no LLM): without the "um" or "uh" the jump cuts remove anyway."""
    words = text.split()
    while words and re.sub(r"[^\w]", "", words[0].lower()) in FILLERS:
        words.pop(0)
    t = " ".join(words)[:60].rstrip(" ,.")
    return t[:1].upper() + t[1:]


def _snap(sents: list[Sentence], start: float, end: float, min_s: float, max_s: float) -> tuple[float, float] | None:
    if not sents:
        return None
    i = min(range(len(sents)), key=lambda k: abs(sents[k].start - start))
    j = min(range(i, len(sents)), key=lambda k: abs(sents[k].end - end))
    while j > i and sents[j].end - sents[i].start > max_s:
        j -= 1
    while j + 1 < len(sents) and sents[j].end - sents[i].start < min_s:
        j += 1
    s0, e0 = sents[i].start, sents[j].end
    if e0 - s0 > max_s:  # one very long sentence: cut it
        e0 = s0 + max_s
    return (s0, e0) if e0 - s0 >= min(min_s, 5.0) else None


def pick_clips(llm, sents: list[Sentence], settings: Settings, count: int, min_s: int = 20,
               max_s: int = 58, brief: str = "") -> list[ClipPlan]:
    """Return up to count ranked candidate moments (not necessarily all rendered)."""
    if llm is None:  # offline: windows spread evenly over the whole video, each starting on a sentence
        plans, nxt = [], 0
        if not sents:
            return plans
        t0, step = sents[0].start, (sents[-1].end - sents[0].start) / max(count, 1)
        for k in range(count):
            i = next((n for n in range(nxt, len(sents)) if sents[n].start >= t0 + k * step), None)
            if i is None:
                break
            j = i
            while j + 1 < len(sents) and sents[j].end - sents[i].start < min_s:
                j += 1
            end = min(sents[j].end, sents[i].start + max_s)
            if end - sents[i].start < min(min_s, 5.0):
                continue
            plans.append(ClipPlan(sents[i].start, end, title=_title_from(sents[i].text) or f"Clip {k + 1}",
                                  hook_text=""))
            nxt = j + 1
        return plans
    system, user = clip_prompt(sents, count, min_s, max_s, settings, brief)
    data = llm.complete_json(system, user, CLIP_SCHEMA, max_tokens=6000, temperature=0.3)  # judge, don't invent
    plans: list[ClipPlan] = []
    for c in data.get("clips") or []:
        try:
            snapped = _snap(sents, float(c["start"]), float(c["end"]), min_s, max_s)
        except (KeyError, TypeError, ValueError):
            continue
        if not snapped:
            continue
        s0, e0 = snapped
        if any(s0 < p.end and e0 > p.start for p in plans):
            continue
        plans.append(ClipPlan(s0, e0, str(c.get("title") or "").strip(), str(c.get("hook_text") or "").strip(),
                              str(c.get("description") or "").strip(), list(c.get("hashtags") or []),
                              list(c.get("tags") or []), int(c.get("score") or 0), str(c.get("reason") or "")))
    plans.sort(key=lambda p: p.score or 0, reverse=True)
    return plans[:count]


# ------------------------------------------------------------------------------------------ face framing
def _face_detector(settings: Settings):
    """A function image -> [(centre x, centre y, area, score)], or None without OpenCV. Prefers the YuNet
    neural detector (its 230 KB model is downloaded once into data/models); falls back to OpenCV 4's
    Haar cascade."""
    try:
        import cv2
    except ImportError:
        return None
    if hasattr(cv2, "FaceDetectorYN"):
        model = settings.data_path / "models" / "face_detection_yunet_2023mar.onnx"
        try:
            if not model.exists() or model.stat().st_size < 100_000:
                download(YUNET_URL, model, timeout=60, max_bytes=5 * 1024 * 1024)
            det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 20)

            def yunet(img):
                h, w = img.shape[:2]
                det.setInputSize((w, h))
                _, faces = det.detect(img)
                return [(float(f[0] + f[2] / 2), float(f[1] + f[3] / 2), float(f[2] * f[3]), float(f[-1]))
                        for f in (faces if faces is not None else [])]
            return yunet
        except Exception as e:
            log.warning("Face detector model unavailable (%s)", e)
    if hasattr(cv2, "CascadeClassifier"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def haar(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return [(x + w / 2, y + h / 2, float(w * h), 1.0)
                    for x, y, w, h in cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))]
        return haar
    return None


def face_samples(settings: Settings, source: Path, start: float, dur: float, work: Path,
                 every: float = 0.5) -> list[tuple[float, float | None]] | None:
    """(time in the clip, horizontal centre of the main face as a fraction of the width) about twice a second.
    The main face is the biggest confident one. None when no detector is available."""
    detect = _face_detector(settings)
    if detect is None:
        return None
    import cv2
    frames = work / "faces"
    frames.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(source), "-vf",
                f"fps={1 / every:.4f},scale=640:-2", str(frames / "f%04d.jpg")], label="face samples")
    out: list[tuple[float, float | None]] = []
    for k, f in enumerate(sorted(frames.glob("f*.jpg"))):
        img = cv2.imread(str(f))
        if img is None:
            continue
        faces = detect(img)
        main = max(faces, key=lambda c: c[2] * c[3], default=None)
        out.append(((k + 0.5) * every, main[0] / img.shape[1] if main else None))
    return out


def plan_face_crop(samples: list[tuple[float, float | None]], dur: float, jump: float = 0.12,
                   min_seconds: float = 1.0) -> list[tuple[float, float]]:
    """[(from time, face centre x)]: one steady framing per camera shot. A new framing starts when the face
    moves more than ``jump`` of the width (a camera cut, or the other speaker); shorter than ``min_seconds``
    and it joins the framing before it, so detection noise never makes the picture jitter."""
    known = _smooth([(t, x) for t, x in samples if x is not None])
    if not known:
        return []
    segs: list[list] = []
    for t, x in known:
        if segs and abs(x - median(segs[-1][1])) <= jump:
            segs[-1][1].append(x)
        else:
            segs.append([t, [x]])
    merged: list[list] = []
    for i, (t0, xs) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else dur
        if merged and end - t0 < min_seconds:
            merged[-1][1].extend(xs)
        else:
            merged.append([t0, list(xs)])
    return [(0.0 if i == 0 else round(t0 - 0.25, 2), round(median(xs), 4)) for i, (t0, xs) in enumerate(merged)]


def _smooth(known: list[tuple[float, float]], radius: int = 2) -> list[tuple[float, float]]:
    """A running median over neighbouring samples: one stray detection (the other speaker for a moment, a
    poster on the wall) is outvoted instead of starting a framing of its own."""
    xs = [x for _, x in known]
    out = []
    for i, (t, x) in enumerate(known):
        win = sorted(xs[max(0, i - radius):i + radius + 1])
        n = len(win)
        if n % 2:
            m = win[n // 2]
        else:  # even window: of the two middle values, the one nearer this sample
            lo, hi = win[n // 2 - 1], win[n // 2]
            m = lo if abs(lo - x) <= abs(hi - x) else hi
        out.append((t, m))
    return out


def face_crop_filter(settings: Settings, src_w: int, src_h: int, plan: list[tuple[float, float]]) -> str | None:
    """A crop that keeps the face in frame, switching framing at the planned times. None when the source is
    not wider than the output (nothing to choose) or no face was found."""
    W, H = settings.width, settings.height
    if not plan or not src_w or not src_h or src_w / src_h <= W / H + 0.01:
        return None
    cw, ch = int(src_h * W / H) // 2 * 2, src_h // 2 * 2

    def left(c: float) -> int:
        return max(0, min(src_w - cw, int(c * src_w - cw / 2))) // 2 * 2
    expr = str(left(plan[-1][1]))
    for i in range(len(plan) - 2, -1, -1):
        expr = f"if(lt(t,{plan[i + 1][0]:.2f}),{left(plan[i][1])},{expr})"
    return f"crop={cw}:{ch}:x='{expr}':y=0,scale={W}:{H}:flags=lanczos,setsar=1"


# ------------------------------------------------------------------------------------------ editing
# hesitation sounds Whisper writes down (English, Spanish, German, French, Portuguese)
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "mm", "mhm", "eh", "ehm", "äh", "ähm", "euh", "hum"}


def _filler(word: Word) -> bool:
    return re.sub(r"[^\w]", "", word.text.lower()) in FILLERS


def keep_ranges(words: list[Word], dur: float, fps: int, *, max_pause: float = 0.6, pad: float = 0.15,
                fillers: bool = True) -> list[tuple[float, float]]:
    """Jump cuts: the parts of a clip to keep, as (start, end) seconds on frame boundaries. Pauses longer
    than ``max_pause`` shrink to about 2 x ``pad`` and filler sounds ("um", "uh") are cut together with
    the pauses around them. Cutting on frame boundaries lets the picture and the sound be cut at exactly
    the same instants, so lips stay in sync however many cuts there are."""
    if not words:
        return [(0.0, dur)]
    cuts: list[tuple[float, float]] = []
    spoken = [w for w in words if not (fillers and _filler(w))]
    if not spoken:
        return [(0.0, dur)]
    if spoken[0].start > 2 * pad:
        cuts.append((0.0, spoken[0].start - pad))
    for a, b in zip(spoken, spoken[1:]):
        gap = b.start - a.end
        has_filler = fillers and any(_filler(w) and a.end <= w.start < b.start for w in words)
        if gap > max_pause or (has_filler and gap > 2 * pad):
            cuts.append((a.end + pad, b.start - pad))
    tail = min(dur, spoken[-1].end + 0.25)
    if dur - tail > 1.0 / fps:
        cuts.append((tail, dur))
    if not cuts:
        return [(0.0, dur)]

    def snap(t: float) -> float:
        return round(round(t * fps) / fps, 6)
    end = round(int(dur * fps + 1e-6) / fps, 6)  # never past the last frame of the clip
    keep, t = [], 0.0
    for c0, c1 in sorted(cuts):
        c0, c1 = snap(c0), min(snap(c1), end)
        if c1 - c0 < 1.0 / fps or c0 < t:
            continue
        if c0 - t >= 2.0 / fps:
            keep.append((t, c0))
        t = c1
    if end - t >= 2.0 / fps:
        keep.append((t, end))
    return keep or [(0.0, dur)]


def retime(words: list[Word], keep: list[tuple[float, float]]) -> list[Word]:
    """Word times after the jump cuts; words that fell in a cut (fillers) are dropped."""
    out, removed, k = [], 0.0, 0
    for w in words:
        while k < len(keep) and w.start >= keep[k][1]:
            k += 1
        if k >= len(keep):
            break
        a, b = keep[k]
        if w.start < a - 0.02:
            continue
        removed = a - sum(y - x for x, y in keep[:k])
        out.append(Word(w.text, round(max(0.0, w.start - removed), 3), round(min(b, w.end) - removed, 3),
                        w.speaker))
    return out


def select_expr(keep: list[tuple[float, float]], fps: int) -> str:
    """An ffmpeg select/aselect expression true for frames in the kept parts. Half a frame of tolerance
    so a frame that starts exactly on a boundary is decided the same way for picture and sound."""
    h = 0.5 / fps
    return "+".join(f"gte(t,{a - h:.4f})*lt(t,{b - h:.4f})" for a, b in keep)


def zoom_plan(words: list[Word], dur: float, *, zoom: float = 1.12, min_seconds: float = 2.0) -> list[tuple[float, float]]:
    """[(from time, zoom)]: the framing punches in and out at sentence starts, at most every
    ``min_seconds``, the hard-cut zoom editors use to keep a talking head lively."""
    plan, level, last = [(0.0, 1.0)], 1.0, 0.0
    for prev, w in zip(words, words[1:]):
        new_sentence = prev.text.rstrip()[-1:] in ".!?" or w.start - prev.end > 0.45
        if new_sentence and w.start - last >= min_seconds and dur - w.start >= min_seconds:
            level = zoom if level == 1.0 else 1.0
            t = max(0.0, w.start - 0.05)
            plan.append((round(t, 3), level))
            last = t
    return plan


def zoom_filter(settings: Settings, plan: list[tuple[float, float]]) -> str:
    """zoompan with a piecewise-constant zoom (no slow drift): each change is a clean cut. The zoom is
    centred a little above the middle, where faces usually are."""
    if len(plan) < 2:
        return ""
    expr = f"{plan[-1][1]:.3f}"
    for i in range(len(plan) - 2, -1, -1):
        expr = f"if(lt(it,{plan[i + 1][0]:.3f}),{plan[i][1]:.3f},{expr})"
    W, H, F = settings.width, settings.height, settings.fps
    return (f",zoompan=z='{expr}':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*0.4':d=1:s={W}x{H}:fps={F},"
            f"setsar=1")


# ------------------------------------------------------------------------------------------ rendering
def crop_filter(settings: Settings, mode: str, face: str | None = None) -> str:
    W, H = settings.width, settings.height
    cover = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1"
    if mode == "face" and face:
        return face
    if mode == "blur":  # whole frame in the middle, a blurred copy fills the rest
        return (f"split[a][b];[a]{cover},boxblur=24:3,eq=brightness=-0.10[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    return cover


def render_clip(settings: Settings, source: Path, plan: ClipPlan, words: list[Word], folder: Path,
                crop: str = "center", src_size: tuple[int, int] | None = None,
                has_audio: bool = True) -> tuple[Path, list, float]:
    s = replace(settings, music_volume=0.0, transition="none", ken_burns=False)
    work = folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    dur = round(plan.duration + 0.25, 3)
    F = s.fps
    face = None
    if crop == "face":
        if src_size is None:
            info = ffmpeg.probe(source)
            src_size = (info["width"], info["height"])
        samples = face_samples(s, source, plan.start, dur, work)
        if samples is None:
            log.warning("Face framing needs OpenCV (pip install opencv-python-headless); cropping the centre")
        else:
            framing = plan_face_crop(samples, dur)
            face = face_crop_filter(s, *src_size, framing)
            log.info("Face framing: %s", f"{len(framing)} shot(s)" if face else "no face found, cropping the centre")
    clip_words = [Word(w.text.strip(), round(w.start - plan.start, 3), round(w.end - plan.start, 3))
                  for w in words if plan.start - 0.05 <= w.start < plan.end]
    keep = keep_ranges(clip_words, dur, F, max_pause=s.clip_max_pause) if s.clip_jump_cuts else [(0.0, dur)]
    cut = len(keep) > 1 or keep[0] != (0.0, dur)
    picked, sound = "", ""
    if cut:
        expr = select_expr(keep, F)
        picked = f",select='{expr}',setpts=N/({F}*TB)"
        # audio frames exactly one video frame long, so both are kept or dropped together
        sound = f"asetnsamples=n={round(48000 / F)}:p=0,aselect='{expr}',asetpts=N/SR/TB,"
        clip_words = retime(clip_words, keep)
        new_dur = round(sum(b - a for a, b in keep), 3)
        log.info("Jump cuts: %d cut(s), %.1fs -> %.1fs", len(keep) - 1 + (keep[0][0] > 0), dur, new_dur)
        dur = new_dur
    zoom = zoom_filter(s, zoom_plan(clip_words, dur)) if s.clip_zoom and crop != "blur" else ""
    grade = _grade(s)
    seg = work / "seg00.mp4"
    ffmpeg.run(["-ss", f"{plan.start:.3f}", "-i", str(source), "-t", f"{plan.duration + 0.25:.3f}", "-an",
                "-vf", f"{crop_filter(s, crop, face)},fps={F}{grade}{picked}{zoom}", *_encode_args(s, final=False),
                "-frames:v", str(max(2, round(dur * F))), str(seg)], label="clip video")
    voice = work / "voice.wav"
    if has_audio:
        ffmpeg.run(["-ss", f"{plan.start:.3f}", "-i", str(source), "-t", f"{plan.duration + 0.25:.3f}", "-vn",
                    "-af", f"aresample=48000,{sound}aformat=channel_layouts=mono", "-ar", "48000", str(voice)],
                   label="clip audio")
    else:
        ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
                    "-t", f"{dur:.3f}", "-ar", "48000", "-ac", "1", str(voice)], label="silent clip audio")
    chunks = caption_chunks(clip_words, s.caption_max_words, total=dur)
    overlay = build_overlays(s, chunks, dur, plan.hook_text, work)
    video = folder / ("short.mp4" if s.vertical else "video.mp4")
    compose(s, [seg], [dur], voice, dur, overlay, video, work, None)
    return video, chunks, dur


def make_clips(studio, src: str, *, count: int = 3, crop: str = "center", upload: bool | None = None,
               min_seconds: int = 20, max_seconds: int = 58, brief: str = "",
               idea_count: int = 10, jump_cuts: bool | None = None, zoom: bool | None = None,
               edit_style: str = "auto") -> list:
    """Returns one pipeline.Result per clip. ``brief``: what to look for ("the funniest moments", "every
    tip about sleep"); without an LLM it is ignored."""
    from .history import now_iso
    from .pipeline import Result, final_description, final_title

    s = studio.s
    # Studio controls must affect the renderer, not just the browser state.
    s = replace(s,
                clip_jump_cuts=s.clip_jump_cuts if jump_cuts is None else bool(jump_cuts),
                clip_zoom=s.clip_zoom if zoom is None else bool(zoom))
    upload = s.upload if upload is None else upload
    count = max(1, min(10, int(count)))
    idea_count = max(count, min(10, int(idea_count)))
    edit_style = edit_style if edit_style in {"auto", "cinematic", "hype", "manga"} else "auto"
    crop = crop if crop in CROPS else "center"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    src_name = Path(src).stem if not re.match(r"^https?://", src) else "video"
    scratch = s.out_path / f".clip-{stamp}-{slugify(src_name, 24)}"
    results = []
    try:
        studio._step("topic", "reading the video")
        source = fetch_source(src, scratch)
        info = ffmpeg.probe(source)
        if not info["has_video"]:
            raise PermanentError(f"{src}: no video stream was found")
        has_audio = bool(info.get("has_audio"))
        words: list[Word] = []
        if has_audio:
            studio._step("voice", "transcribing audio (optional)")
            try:
                words = transcribe(source, scratch, s)
            except Exception as e:
                log.warning("Speech analysis unavailable; using visual-only clip selection: %s", str(e)[:180])
        else:
            log.info("Source has no audio; continuing with visual-only clipping")
        sents = sentences(words) if words else []
        duration = float(info.get("duration") or 0)
        if duration <= 0:
            raise RuntimeError("could not determine the video's duration")
        studio._step("script", "analyzing scenes and finding usable moments")
        native_features = []
        native_context = {}
        try:
            native_features = native_visual_features(source, duration, scratch)
            if native_features:
                native_context = enrich_native_visual_context(source, duration, scratch, native_features)
                apply_visual_context_scores(native_features, native_context)
                strong_changes = sum(1 for frame in native_features if frame["change"] >= 0.34)
                log.info("Native vision scanned %d frames: %d strong changes, %d face frames, %d repeated face tracks, %d OCR frames",
                         len(native_features), strong_changes, native_context.get("face_frames", 0),
                         native_context.get("face_tracks", 0), native_context.get("ocr_frames", 0))
        except Exception as e:
            log.warning("Native visual feature scan unavailable; continuing with other analysis: %s", str(e)[:180])
        try:
            llm = studio.llm
        except LLMConfigError as e:
            log.warning("No LLM is configured; using non-AI fallback moments (%s)", str(e).split(".")[0])
            llm = None
        analysis_mode = "visual-fallback"
        plans = []
        if llm is not None:
            try:
                visual_frames = sample_visual_frames(source, duration, scratch, limit=12)
                if visual_frames:
                    plans = pick_visual_clips(llm, visual_frames, duration, sents, s, idea_count,
                                              min_seconds, min(max_seconds, 170), brief)
                    if plans:
                        analysis_mode = "multimodal-ranked"
                        log.info("Vision model ranked %d moments using %d sampled frames%s",
                                 len(plans), len(visual_frames), " and transcript" if sents else "")
            except Exception as e:
                log.warning("Visual AI analysis unavailable; falling back to transcript analysis: %s", str(e)[:220])
        if not plans and native_features:
            plans = rank_native_visual_moments(native_features, duration, idea_count,
                                               min_seconds, min(max_seconds, 170))
            if plans:
                analysis_mode = "native-pixel-ranked"
                log.info("Native pixel analysis ranked %d visual moments without an AI model", len(plans))
        if not plans and sents:
            try:
                plans = pick_clips(llm, sents, s, idea_count, min_seconds, min(max_seconds, 170), brief)
                analysis_mode = "transcript-ranked" if llm is not None else "transcript-fallback"
            except Exception as e:
                log.warning("Transcript AI ranking failed; using evenly spaced transcript moments: %s",
                            str(e)[:220])
                plans = pick_clips(None, sents, s, idea_count, min_seconds, min(max_seconds, 170), brief)
                analysis_mode = "transcript-fallback"
        if not plans:
            # Non-AI fallback: still creates usable cuts, but never labels them as AI-selected.
            upper = min(float(max_seconds), duration)
            lower = min(float(min_seconds), upper)
            if upper < 1:
                raise RuntimeError("video is too short to clip")
            span = min(upper, max(lower, min(30.0, duration)))
            latest = max(0.0, duration - span)
            n = min(idea_count, max(1, int(duration // max(span * 0.55, 1)) + 1))
            starts = [latest * k / max(n - 1, 1) for k in range(n)]
            plans = [ClipPlan(round(start, 3), round(min(duration, start + span), 3),
                              f"Visual moment {i + 1}",
                              reason="Non-AI evenly spaced fallback; visual understanding unavailable")
                     for i, start in enumerate(starts)]
            analysis_mode = "visual-fallback"
            log.warning("Using evenly distributed fallback moments; no visual or transcript ranking was available.")
        if not plans:
            raise RuntimeError("no usable moments found")
        ideas = [{
            "rank": i, "start": round(p.start, 2), "end": round(p.end, 2),
            "duration": round(p.duration, 2), "title": p.title, "hook_text": p.hook_text,
            "score": p.score, "reason": p.reason, "description": p.description,
            "hashtags": p.hashtags or [], "tags": p.tags or [],
        } for i, p in enumerate(plans[:idea_count], 1)]
        plans = plans[:count]
        log.info("Analyzed %d candidate moment(s); rendering the top %d from %s", len(ideas), len(plans), src)
        style_cycle = ("cinematic", "hype", "manga")
        for n, plan in enumerate(plans, 1):
            studio._step("render", f"clip {n}/{len(plans)}")
            t0 = time.time()
            chosen_style = style_cycle[(n - 1) % len(style_cycle)] if edit_style == "auto" else edit_style
            style_settings = {
                "cinematic": {"color_grade": "cinematic", "caption_style": "clean"},
                "hype": {"color_grade": "vivid", "caption_style": "karaoke"},
                "manga": {"color_grade": "manga", "caption_style": "boxed"},
            }[chosen_style]
            clip_settings = replace(s, **style_settings)
            text = " ".join(w.text.strip() for w in words if plan.start <= w.start < plan.end)
            script = normalize({"title": plan.title, "hook_text": plan.hook_text, "description": plan.description,
                                "hashtags": plan.hashtags or [], "tags": plan.tags or [], "category": "entertainment",
                                "scenes": [{"narration": text or plan.title, "search_query": "clip"}]},
                               plan.title or src_name, "story", s.language)
            folder = (s.out_path / f"{stamp}_{slugify(script.title)}-{n}").resolve()
            folder.mkdir(parents=True, exist_ok=True)
            try:
                video, chunks, dur = render_clip(clip_settings, source, plan, words, folder, crop,
                                                 (info["width"], info["height"]), has_audio=has_audio)
                out = ffmpeg.probe(video)
                cover_at = pick_cover(video, folder / "cover.jpg", folder / "work", total=dur, until=min(3.0, dur * 0.4))
                (folder / "captions.srt").write_text(srt(chunks), encoding="utf-8")
                title, description = final_title(script, s), final_description(script, set(), s)
                (folder / "clip_ideas.json").write_text(
                    json.dumps({"source": src, "analysis": analysis_mode,
                                "rendered_count": len(plans), "ideas": ideas}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                meta = {"title": title, "description": description, "tags": script.tags, "category_id": "24",
                        "clip_ideas": ideas, "clip_analysis": analysis_mode, "edit_style": chosen_style,
                        "language": s.language, "source": "clip", "style": "clip", "clip_of": src,
                        "clip_start": round(plan.start, 2), "clip_end": round(plan.end, 2),
                        "score": plan.score, "review": [plan.reason] if plan.reason else [],
                        "duration": round(out["duration"], 2), "resolution": f"{s.width}x{s.height}",
                        "llm": llm.label if llm else "offline", "created_at": now_iso(), "cover_at": round(cover_at, 2)}
                (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                rid = studio.history.add_video(
                    subject=src, source="clip", topic=script.topic, style="clip", language=s.language, title=title,
                    description=description, tags=script.tags, folder=str(folder), video_path=str(video),
                    duration=out["duration"], llm=meta["llm"], voice="original audio", status="rendered",
                    score=plan.score)
                r = Result(True, title, folder, video, "rendered", seconds=time.time() - t0,
                           record_id=rid, score=plan.score, language=s.language, reason=plan.reason)
                log.info("Clip %d: %s (%.0fs from %s)", n, title, dur, _ts(plan.start))
                if upload:
                    studio.upload_record(rid, r)
                results.append(r)
            except PermanentError:
                raise
            except Exception as e:
                log.error("Clip %d failed: %s", n, e)
                results.append(Result(False, plan.title, folder, status="failed", error=str(e)))
            finally:
                if not os.getenv("KEEP_WORK"):
                    shutil.rmtree(folder / "work", ignore_errors=True)
        studio._step("done")
        return results
    finally:
        if not os.getenv("KEEP_WORK"):
            shutil.rmtree(scratch, ignore_errors=True)


def _ts(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"
