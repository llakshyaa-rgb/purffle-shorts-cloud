"""The production line: topic -> script -> review -> voice -> footage -> captions -> render -> upload.

``Studio.draft`` stops after the script (so it can be read and edited first); ``Studio.make`` runs the
whole line, optionally followed by translated copies that reuse the same footage."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import analytics, crosspost, ffmpeg, notify, seo, sfx, topics, tts, youtube
from .config import Settings
from .creative_director import apply_to_script
from .creative_director import direct as direct_scenes
from .creative_director import notes_markdown as director_notes_markdown
from .history import History, now_iso
from .llm import LLM, LLMConfigError
from .media import MediaItem, Visuals, background_item, relevance
from .overlays import EMOJI_CREDIT, ChatMessage, build_overlays
from .render import pick_cover, render_video, split_sizes
from .script import (
    AUTO_STYLES,
    CARD_STYLES,
    DEFAULT_CAST,
    LANGUAGES,
    MULTI_SPEAKER,
    Script,
    load_script,
    localize_metadata,
    review_script,
    translate_script,
    write_script,
)
from .timing import SceneTiming, caption_chunks, scene_timeline, split_shots, srt, tokenize
from .topics import Topic, pick_topic
from .utils import Cancelled, check_cancel, redact, slugify, truncate_words

log = logging.getLogger("purffle")

END_PAD = 0.6          # seconds of breathing room after the last word


def _can_adaptively_reuse(item: MediaItem) -> bool:
    """Only replace failed generated placeholders; preserve deliberate gradient art direction."""
    return item.source == "generated" and not bool(item.extra.get("intentional_gradient"))


REUSABLE_SOURCES = {"generated", "local", "library", "background", "openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare"}
AI_SOURCES = {"openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare"}

# Rough share of the work done when each stage starts (for progress bars before any timing is learned).
STAGES = {"topic": 2, "script": 8, "review": 20, "voice": 32, "footage": 45, "captions": 62, "render": 72,
          "check": 92, "upload": 95, "translate": 97, "done": 100}
ORDER = list(STAGES)
# Seconds each stage takes for a 40-second video until this machine's own timings are known.
DEFAULT_SECONDS = {"topic": 1.0, "script": 15.0, "review": 25.0, "voice": 6.0, "footage": 12.0, "captions": 3.0,
                   "render": 20.0, "check": 1.0, "upload": 30.0, "translate": 70.0}

ProgressFn = Callable[[str, int, str], None]


class StageClock:
    """How long each stage takes on this machine: a moving average per stage, scaled to the video's length
    and kept in the history database, so progress bars and ETAs follow real time instead of a fixed table."""

    KEY = "stage_seconds"

    def __init__(self, history: History, target_seconds: int):
        self.history = history
        self.scale = max(0.25, target_seconds / 40)
        try:
            self.avg: dict[str, float] = json.loads(history.get_meta(self.KEY) or "{}")
        except ValueError:
            self.avg = {}

    @property
    def learned(self) -> bool:
        return {"voice", "render"} <= set(self.avg)

    def expected(self, stage: str) -> float:
        if stage == "done":
            return 0.0
        return self.avg.get(stage, DEFAULT_SECONDS.get(stage, 2.0)) * self.scale

    def record(self, stage: str, seconds: float) -> None:
        if stage not in DEFAULT_SECONDS or seconds <= 0:
            return
        raw = seconds / self.scale
        self.avg[stage] = round(raw if stage not in self.avg else 0.7 * self.avg[stage] + 0.3 * raw, 2)
        try:
            self.history.set_meta(self.KEY, json.dumps(self.avg, sort_keys=True))
        except Exception:  # timing statistics are a nicety
            log.debug("could not save stage timings", exc_info=True)


@dataclass
class Result:
    ok: bool
    title: str = ""
    folder: Path | None = None
    video: Path | None = None
    status: str = ""
    youtube_id: str | None = None
    publish_at: str | None = None
    seconds: float = 0.0
    error: str = ""
    record_id: int | None = None
    score: int | None = None
    language: str = ""
    variants: list[Result] = field(default_factory=list)
    reason: str = ""          # clips: why the AI picked this moment
    posts: dict = field(default_factory=dict)  # platform -> {status, url, id, error} beyond YouTube


def quota_day_start() -> str:
    """YouTube's daily quota resets at midnight Pacific time."""
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    start = pt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).isoformat(timespec="seconds")


def seconds_until_quota_reset() -> float:
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (pt.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp() + 86400
    return max(60.0, nxt - pt.timestamp() + 120)


def _future(iso: str | None) -> datetime | None:
    """A stored publish time that is still ahead, so a retry posts the other networks at the same slot."""
    if not iso:
        return None
    try:
        t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return t if t > datetime.now(timezone.utc) else None


def final_title(script: Script, settings: Settings | None = None, text: str | None = None) -> str:
    t = script.title if text is None else text
    if settings is not None and not settings.vertical:
        return truncate_words(t, 100)  # a 16:9 video is not a Short
    return f"{t} #shorts" if len(t) + 8 <= 100 else truncate_words(t, 100)


def final_description(script: Script, credits: set[str], settings: Settings, text: str | None = None,
                      extra: list[str] | None = None) -> str:
    body = (script.description if text is None else text).strip()
    parts = [body] if body else []
    tags = script.hashtags if settings.vertical else [h for h in script.hashtags if h.lower() != "#shorts"]
    parts.append(" ".join(tags))
    if settings.credit_footage and credits:
        parts.append("Footage: " + ", ".join(sorted(credits)))
    parts.extend(extra or [])
    return "\n\n".join(p for p in parts if p)


def _unique_folder(root: Path, name: str) -> Path:
    folder = root / name
    n = 2
    while folder.exists():
        folder = root / f"{name}-{n}"
        n += 1
    return folder


def _media_for(shots, media_map: dict[tuple[int, int], MediaItem]) -> list[MediaItem]:
    """Reuse the original video's footage for a translated copy. Its scenes may merge or split into shots
    differently, so each shot takes the closest (scene, shot) of the original."""
    keys = sorted(media_map)
    out = []
    for st in shots:
        same = [k for k in keys if k[0] == st.index]
        if same:
            k = max([k for k in same if k[1] <= st.part], default=same[0])
        else:
            k = max([k for k in keys if k[0] <= st.index], default=keys[0])
        out.append(media_map[k])
    return out


class Studio:
    def __init__(self, settings: Settings, progress: ProgressFn | None = None):
        self.s = settings
        self.history = History(settings.data_path / "history.db")
        self._llm: LLM | None = None
        self.progress = progress
        self._stats_synced = 0.0
        self.clock = StageClock(self.history, settings.target_seconds)
        self._learn = False            # only whole videos teach the clock (clips use the stages differently)
        self._quiet = False            # translated copies run as one "translate" stage
        self._plan: list[str] = []     # the stages this job will go through
        self._stage, self._stage_t0, self._frac, self._msg = "", 0.0, 0.0, ""
        self._t_start, self._pct = 0.0, 0

    # ------------------------------------------------------------------ progress, ETA, cancel
    def _begin(self, plan: list[str]) -> None:
        self._plan, self._learn = plan, True
        self._stage, self._t_start, self._pct = "", time.time(), 0

    def _step(self, stage: str, message: str = "") -> None:
        check_cancel()
        if self._quiet:
            return
        now = time.time()
        # Demo runs write instantly, so they must not teach the clock how long the AI stages take.
        writing = self._stage in ("topic", "script", "review", "translate")
        if self._learn and self._stage and self._stage != stage and not (self.s.offline and writing):
            self.clock.record(self._stage, now - self._stage_t0)
        self._stage, self._stage_t0, self._frac, self._msg = stage, now, 0.0, message
        self._report()
        if stage == "done":  # the job is over: the next draft or make starts its own plan
            self._learn, self._plan = False, []

    def _sub(self, frac: float, message: str = "") -> None:
        """Progress inside the current stage (0..1), e.g. render frames written."""
        if self._quiet:
            return
        self._frac = max(self._frac, min(1.0, frac))
        if message:
            self._msg = message
        self._report()

    def eta(self) -> float | None:
        """Seconds until the job is done, from this machine's stage timings, or None when unknown."""
        if not self._stage or self._stage == "done" or not self._plan:
            return None
        cur = self._stage
        spent = time.time() - self._stage_t0
        exp = self.clock.expected(cur)
        left = exp * (1 - self._frac) if self._frac > 0.02 else max(0.15 * exp, exp - spent)
        after = self._plan[self._plan.index(cur) + 1:] if cur in self._plan else []
        return round(left + sum(self.clock.expected(st) for st in after), 1)

    def _report(self) -> None:
        stage = self._stage
        if stage == "done":
            pct = 100
        elif self.clock.learned and self._plan and (eta := self.eta()) is not None:
            spent = time.time() - self._t_start
            pct = int(99 * spent / max(0.1, spent + eta))
        else:
            nxt = next((STAGES[x] for x in ORDER[ORDER.index(stage) + 1:]), 100) if stage in STAGES else 0
            pct = int(STAGES.get(stage, 0) + self._frac * (nxt - STAGES.get(stage, 0)))
        self._pct = max(self._pct, pct)
        if self.progress:
            try:
                self.progress(stage, self._pct, self._msg)
            except Exception:  # a progress display must never break a video
                log.debug("progress callback failed", exc_info=True)

    @property
    def llm(self) -> LLM | None:
        if self.s.offline:
            return None
        if self._llm is None:
            self._llm = LLM(self.s)
            log.info("LLM: %s%s", self._llm.label,
                     f" (fallbacks: {', '.join(str(p) for p in self._llm.providers[1:])})"
                     if len(self._llm.providers) > 1 else "")
        return self._llm

    # ------------------------------------------------------------------ voice with fallbacks
    def _speak(self, script: Script, work: Path, s: Settings) -> tts.Speech:
        engines = [s.tts_engine]
        if s.tts_engine != "edge":
            engines.append("edge")
        if s.offline:
            engines += ["system", "silent"]
        lines = [(sc.speaker, sc.narration) for sc in script.scenes] if (
            script.multi_speaker or script.style in MULTI_SPEAKER) else None
        last: Exception | None = None
        for eng in dict.fromkeys(engines):
            try:
                cfg = s if eng == s.tts_engine else replace(s, tts_engine=eng, tts_voice="", tts_voice_b="")
                if lines:
                    return tts.synthesize_lines(lines, work / "voice", cfg)
                return tts.synthesize(script.narration, work, cfg)
            except Exception as e:
                last = e
                log.warning("Voice engine '%s' failed: %s", eng, e)
        raise RuntimeError(f"All voice engines failed: {last}")

    # ------------------------------------------------------------------ script only
    def draft(self, topic: str | None = None, *, source: str | None = None, style: str | None = None,
              url: str | None = None, document: str | Path | None = None,
              creative_brief: str = "") -> tuple[Script, Topic, str]:
        """Pick a topic, write the script and let the Script Doctor improve it. Nothing is rendered."""
        s = self.s
        standalone = not self._plan  # a draft on its own (Studio "Draft"), not the first half of make()
        if standalone:
            self._begin(["topic", "script"] + (["review"] if not s.offline and s.script_review != "off" else [])
                        + ["done"])
        self._step("topic")
        if url:
            pick = topics.from_url(url)
        elif document:
            pick = topics.from_document(document)
        else:
            pick = pick_topic(s, self.history, explicit=topic, source=source)
        if topic and (url or document):
            pick.subject = f"{topic} (source: {pick.subject})"
        log.info("Topic [%s]: %s", pick.source, pick.subject)
        try:
            self._step("script", pick.subject)
            insights = analytics.insights(self.history) if s.learn_from_stats else ""
            style = style or pick.style or None
            if (not style and s.learn_from_stats and s.style in ("", "auto")
                    and pick.source not in ("trending", "wikipedia", "reddit", "rss", "url", "document")):
                style = analytics.choose_style(self.history, sorted(set(AUTO_STYLES)))
                if style:
                    log.info("Format from the channel's results: %s", style)
            if s.seo_keywords and self.llm is not None and not s.offline and not pick.keywords:
                pick.keywords = seo.search_phrases(pick.subject, s.language, s.trends_geo)
                if pick.keywords:
                    log.info("YouTube searches: %s", "; ".join(pick.keywords[:4]))
            script = write_script(self.llm, pick.subject, s, source=pick.source, avoid=self.history.recent_titles(),
                                  context=pick.context, style=style, insights=insights,
                                  keywords=seo.keywords_block(pick.keywords), creative_brief=creative_brief)
            if self.llm is not None and s.script_review != "off":
                self._step("review", "Script Doctor")
                script = review_script(self.llm, script, s)
        except BaseException:
            if pick.idea_id:
                self.history.release_idea(pick.idea_id)
            raise
        if standalone:
            self._step("done", script.title)
        return script, pick, self.llm.label if self.llm else "offline"

    # ------------------------------------------------------------------ one video
    def make(self, topic: str | None = None, *, source: str | None = None, upload: bool | None = None,
             style: str | None = None, script_file: str | Path | None = None, script: Script | None = None,
             url: str | None = None, document: str | Path | None = None,
             languages: list[str] | None = None, creative_brief: str = "") -> Result:
        t0 = time.time()
        s = self.s
        upload = s.upload if upload is None else upload
        languages = s.also_languages if languages is None else languages
        state: dict = {}
        pick: Topic | None = None
        writes = script is None and not script_file
        reviews = writes and not s.offline and s.script_review != "off"
        self._begin((["topic", "script"] if writes else []) + (["review"] if reviews else [])
                    + ["voice", "footage", "captions", "render", "check"] + (["upload"] if upload else [])
                    + ["translate"] * len([x for x in dict.fromkeys(languages or []) if x != s.language]) + ["done"])
        try:
            if script is not None or script_file:
                if script is None:
                    script = load_script(script_file, s)
                pick, writer = Topic(script.topic, "script"), ("script file" if script_file else "editor")
                log.info("Script from %s (%d scenes): %s", script_file or "editor", len(script.scenes), script.title)
            else:
                script, pick, writer = self.draft(topic, source=source, style=style, url=url, document=document,
                                                creative_brief=creative_brief)
            return self.produce(script, pick, writer, upload=upload, t0=t0, languages=languages, state=state)
        except (youtube.NotAuthorized, LLMConfigError, KeyboardInterrupt):
            raise  # setup problems stop the run instead of failing every video the same way
        except Cancelled:
            if pick is not None and pick.idea_id and not state.get("record_id"):
                self.history.release_idea(pick.idea_id)
            if state.get("folder") and not state.get("record_id"):
                shutil.rmtree(state["folder"], ignore_errors=True)  # nothing usable was made
            log.info("Stopped: the job was cancelled")
            raise
        except Exception as e:
            if pick is not None and pick.idea_id and not state.get("record_id"):
                self.history.release_idea(pick.idea_id)
            log.error("Video failed: %s", redact(e), exc_info=log.isEnabledFor(logging.DEBUG))
            notify.send(s, "failed", title=getattr(script, "title", "") or (pick.subject if pick else ""),
                        detail=redact(e))
            return Result(False, folder=state.get("folder"), status="failed", error=redact(e),
                          seconds=time.time() - t0)

    def produce(self, script: Script, pick: Topic, writer: str, *, upload: bool, t0: float | None = None,
                languages: list[str] | None = None, media_map: dict[tuple[int, int], MediaItem] | None = None,
                credits: set[str] | None = None, parent_id: int | None = None, settings: Settings | None = None,
                state: dict | None = None) -> Result:
        t0 = t0 or time.time()
        state = state if state is not None else {}
        s = settings or self.s
        if script.language and script.language != s.language:
            s = replace(s, language=script.language, tts_voice="", tts_voice_b="")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{slugify(script.title)}" + (f"-{s.language}" if parent_id else "")
        folder = _unique_folder(s.out_path, name).resolve()  # absolute: history outlives cwd
        state["folder"] = folder
        work = folder / "work"
        work.mkdir(parents=True, exist_ok=True)
        (folder / "script.json").write_text(json.dumps(script.to_dict(), indent=2, ensure_ascii=False))
        try:
            self._step("voice", f"{s.tts_engine} voice")
            speech = self._speak(script, work, s)
            total = round(speech.duration + END_PAD, 3)

            # One bounded duration-repair pass: never silently stretch audio or ship a wildly
            # off-target runtime. Keep the same topic/style and ask the script writer for a tighter
            # word budget, then synthesize the revised narration once.
            target = max(1, int(s.target_seconds))
            repair_llm = None
            # A 20% tolerance made a 33-second video pass for a 25-second request.
            # Repair anything more than 8% off target; publish readiness remains a separate gate.
            if abs(total - target) / target > 0.08 and not s.offline:
                try:
                    repair_llm = self.llm
                except LLMConfigError:
                    # A supplied script file is renderable without an LLM; simply skip optional repair.
                    repair_llm = None
            if repair_llm is not None:
                try:
                    log.warning("Narration runtime %.1fs misses %ss target; attempting one script repair",
                                total, target)
                    repaired = write_script(
                        repair_llm, script.topic, s, source=pick.source, style=script.style,
                        context=(f"Original title: {script.title}. Preserve the original factual topic, "
                                 "central reveal, and intended audience."),
                        creative_brief=(
                            f"RUNTIME REPAIR: target {target} seconds, not a loose estimate. "
                            f"The previous narration rendered to {total:.1f} seconds. "
                            f"Write approximately {max(1, round(target * 4.1))} spoken words, with an acceptable range of "
                            f"{max(1, round(target * 3.7))}–{max(1, round(target * 4.5))} words, including the hook and ending. "
                            "Rewrite the narration; do not explain the revision. The prior attempt was too short, so do NOT compress it further. "
                            "Keep enough spoken dialogue to fill the target runtime; never solve timing by using fewer than the minimum "
                            "word count above. Spoken dialogue must be paced naturally and include meaningful reactions and actions, not padding. "
                            "For a 30-second target, aim near 120–130 spoken words unless the selected voice's measured pace proves otherwise. "
                            "Remove only repeated or irrelevant phrases. Preserve the central reveal, causal logic, "
                            "character reactions, and satisfying ending. No filler or CTA. Use short natural spoken sentences, "
                            "and keep each scene visually concrete. Split the full spoken word budget across enough scenes. "
                            "The narration must fit the requested runtime when spoken aloud."
                        ),
                    )
                    repaired = review_script(repair_llm, repaired, s)
                    repaired_speech = self._speak(repaired, work, s)
                    repaired_total = round(repaired_speech.duration + END_PAD, 3)
                    if abs(repaired_total - target) < abs(total - target):
                        script, speech, total = repaired, repaired_speech, repaired_total
                        (folder / "script.json").write_text(
                            json.dumps(script.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        log.info("Runtime repair improved duration to %.1fs (target %ss)", total, target)
                    else:
                        log.warning("Runtime repair did not improve duration (%.1fs -> %.1fs)",
                                    total, repaired_total)
                except Exception as repair_error:
                    log.warning("Runtime repair failed; quality gate will block off-target publishing: %s",
                                redact(repair_error))
            # Do not spend image-generation time on a video that already failed its requested
            # runtime. The repair is best-effort; if the voice still misses by more than 8%,
            # fail here with an explicit duration error before the Creative Director / footage stage.
            # Offline rendering, editor-supplied scripts, and script files are deterministic
            # rendering workflows: do not reject them because their content is intentionally short.
            # Enforce the runtime gate only for generated scripts in online production.
            enforce_runtime = not s.offline and s.tts_engine != "silent" and parent_id is None and writer not in ("script file", "editor")
            if enforce_runtime and abs(total - target) / target > 0.20:
                raise RuntimeError(
                    f"Runtime gate: requested {target}s, generated {total:.1f}s of narration after repair. "
                    "The narration remains more than 20% off target after the bounded repair."
                )
            if enforce_runtime and abs(total - target) / target > 0.08:
                log.warning(
                    "Narration remains %.1fs from the %ss target after repair; continuing to render rather than "
                    "discarding the entire video (within the 20%% hard limit).",
                    abs(total - target), target
                )
            chat = script.style == "chat"
            timeline = scene_timeline([sc.narration for sc in script.scenes], speech.words, total, s.language,
                                      starts=speech.line_starts, min_len=4.5 if chat else 1.2)

            # Run the Scene-Level Creative Director silently during production. Its visual prompts
            # are consumed by media selection, while camera/sound choices are applied downstream.
            try:
                director_llm = self.llm
            except Exception as director_error:
                log.warning("Creative Director using local fallback: %s", redact(director_error))
                director_llm = None
            director_plan = direct_scenes(script, director_llm)
            apply_to_script(script, director_plan)
            director_payload = {
                "version": 1,
                "title": script.title,
                "topic": script.topic,
                "mode": "llm" if director_llm is not None else "local_fallback",
                "scenes": director_plan,
            }
            (folder / "director.json").write_text(
                json.dumps(director_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (folder / "director_notes.md").write_text(
                director_notes_markdown(script, director_plan), encoding="utf-8"
            )
            (folder / "script.json").write_text(
                json.dumps(script.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info("Creative Director: planned %d scenes; notes saved to director_notes.md",
                     len(director_plan))
            # Per-scene rationale stays in director_notes.md and is shown only when requested in Studio.

            # A chat story's picture is mostly the phone screen: more cuts behind it would only cost time.
            shots = split_shots(timeline, speech.words, 0 if chat else s.max_shot_seconds)
            joins = [a.index == b.index for a, b in zip(shots, shots[1:])]
            T = s.transition_seconds if s.transition != "none" else 0.0
            background = background_item(s, total)
            split = background is not None and s.background_layout == "split"
            if background is not None:
                log.info("Background (%s): %s from %.0fs", s.background_layout, background.id,
                         background.extra.get("start", 0))

            self._step("footage", f"{len(timeline)} scenes, {len(shots)} shots")
            if background is not None and not split:
                # The whole Short plays over one continuous stretch of the background video.
                shots, joins, media, credits = [SceneTiming(0, 0.0, total)], [], [background], set()
            elif media_map:
                media = _media_for(shots, media_map)
                credits = set(credits or ())
            else:
                # AI images cost money or are rate-limited: the extra shots of a scene reuse its picture with a
                # new camera move instead of asking for another one.
                wanted = [st for st in shots if st.part == 0]
                jobs = [(script.scenes[st.index].search_query, script.scenes[st.index].image_prompt,
                         st.duration + T) for st in wanted]
                # split layout: the footage fills the top half, so search for the shape of that half
                visuals = Visuals(replace(s, resolution=split_sizes(s)[0]) if split else s,
                                  self.history.used_media())
                got = dict(zip(((st.index, st.part) for st in wanted),
                               visuals.for_scenes(jobs, script.topic, work / "media", progress=self._sub)))
                media = []
                for st in shots:
                    exact = got.get((st.index, st.part))
                    if exact is not None:
                        media.append(exact)
                        continue
                    base = got.get((st.index, 0))
                    # Reuse the scene's first shot only if it is real, usable media.
                    # Never turn a failed later shot into a forced repeat of a generated background.
                    if base is not None and base.source != "generated":
                        media.append(base)
                    else:
                        media.append(MediaItem(
                            "generated", f"gradient-{st.index}-{st.part}", "generated",
                            query=script.scenes[st.index].search_query,
                        ))

                # Adaptive fallback: reuse a previous successful visual only when its metadata
                # strongly matches the failed shot. Prefer fresh media; don't force reuse when
                # the match is weak, and avoid repeating the immediately previous shot unless the
                # semantic match is especially strong.
                for pos, (st, item) in enumerate(zip(shots, media)):
                    # A director-requested gradient is the chosen visual, not a missing asset.
                    # Never replace it with a vaguely related photo from the same scene.
                    if not _can_adaptively_reuse(item):
                        continue
                    scene_query = script.scenes[st.index].search_query
                    ranked_reuse = []
                    for candidate_pos, candidate in enumerate(media):
                        if candidate.source == "generated" or candidate_pos == pos:
                            continue
                        # Never borrow another scene's image just because its metadata shares a
                        # keyword. That creates visibly repeated or misleading shots. Reuse is
                        # allowed only between split shots belonging to the same narrated scene.
                        if shots[candidate_pos].index != st.index:
                            continue
                        score = relevance(candidate, scene_query)
                        if score >= 0.58:
                            ranked_reuse.append((score, candidate_pos, candidate))
                    ranked_reuse.sort(key=lambda row: row[0], reverse=True)
                    for score, candidate_pos, candidate in ranked_reuse:
                        is_adjacent_repeat = abs(candidate_pos - pos) == 1
                        if is_adjacent_repeat and score < 0.76:
                            continue
                        media[pos] = candidate
                        log.info(
                            "Scene %d: adaptive visual reuse (match %.2f from shot %d)",
                            pos + 1, score, candidate_pos + 1,
                        )
                        break

                credits = visuals.credits
            # Carry director choices with each shot so the renderer can use them.
            for st, item in zip(shots, media):
                if 0 <= st.index < len(director_plan):
                    beat = director_plan[st.index]
                    item.extra["director_camera"] = beat.get("camera", "push_in")
                    item.extra["director_visual_mode"] = beat.get("visual_mode", "illustration")
                    item.extra["director_transition"] = beat.get("transition", "cut")
            by_scene = {(st.index, st.part): m for st, m in zip(shots, media)}

            self._step("captions")
            chunks = caption_chunks(speech.words, s.caption_max_words, total=total)
            labels = ({chr(ord("A") + i): name for i, name in enumerate(script.cast[:6])}
                      if script.style in ("dialogue", "fantasy") and script.multi_speaker else None)
            messages = ([ChatMessage(sc.speaker, sc.narration, st) for sc, st in zip(script.scenes, speech.line_starts)]
                        if chat and speech.line_starts else None)
            card, card_until, on_screen = None, 0.0, chunks
            if script.style in CARD_STYLES:
                # The post card shows the title while it is read; captions start with the story.
                n0 = len(tokenize(script.scenes[0].narration, s.language))
                if 0 < n0 < len(speech.words):
                    card_until = min(speech.words[n0 - 1].end + 0.35, total * 0.6)
                    community, user = (script.cast + DEFAULT_CAST["reddit"][len(script.cast):])[:2]
                    card = (community, user, script.scenes[0].narration)
                    on_screen = caption_chunks(speech.words[n0:], s.caption_max_words, total=total)
            emoji = []  # each scene's emoji pops up as the scene starts (not over a chat or a post card)
            for k, (sc, tm) in enumerate(zip(script.scenes, timeline)):
                if sc.emoji and not chat and not (card and k == 0):
                    emoji.append((tm.start + 0.05, min(tm.end, tm.start + 1.65), sc.emoji))
            plan = build_overlays(s, on_screen, total, script.hook_text, work, labels=labels, chat=messages,
                                  contact=script.speaker_name("A"), card=card, card_until=card_until,
                                  emphasis={k for sc in script.scenes for k in sc.emphasis}, emoji=emoji)

            self._step("render", f"{s.width}x{s.height}")
            video = folder / ("short.mp4" if s.vertical else "video.mp4")
            effects = sfx.events_for(scene_starts=[st.start for st, j in zip(shots[1:], joins) if not j],
                                     message_starts=speech.line_starts if messages else None,
                                     hook=bool(plan.hook_until or plan.card_until))
            # Add sparse director-picked accents only when they will not collide with an existing cue.
            if s.sound_effects and s.sfx_volume > 0 and not messages:
                last_directed_sound = -10.0
                for beat, tm in zip(director_plan, timeline):
                    cue = beat.get("sound", "none")
                    if cue in ("hit", "pop", "whoosh") and tm.start - last_directed_sound >= 1.4:
                        if cue == "hit" or not any(abs(t - tm.start) < 0.35 for t, _ in effects):
                            effects.append((max(0.05, tm.start), cue))
                            last_directed_sound = tm.start
            render_video(s, media, [st.duration for st in shots], speech.audio, total, plan, video, work, joins,
                         progress=self._sub, effects=effects, background=background if split else None,
                         narrator_topic=script.topic)

            self._step("check")
            info = ffmpeg.probe(video)
            if (info["width"], info["height"]) != s.resolution or abs(info["duration"] - total) > 0.5:
                raise RuntimeError(f"Render check failed: got {info['width']}x{info['height']} "
                                   f"{info['duration']:.2f}s, expected {s.width}x{s.height} {total:.2f}s")
            cover_at = pick_cover(video, folder / "cover.jpg", work, total=total, avoid=[st.start for st in shots[1:]],
                                  until=card_until or plan.hook_until or min(2.8, total * 0.4))
            (folder / "captions.srt").write_text(srt(chunks), encoding="utf-8")

            title = final_title(script, s)
            extra = [EMOJI_CREDIT] if plan.emoji and s.credit_footage else []
            description = final_description(script, credits, s, extra=extra)
            localizations = {}
            wanted = [x for x in s.localize_metadata if x != s.language]
            if wanted and self.llm is not None and not s.offline:
                try:
                    for lang, v in localize_metadata(self.llm, script, wanted).items():
                        localizations[lang] = {"title": final_title(script, s, v["title"]),
                                               "description": final_description(script, credits, s,
                                                                                v["description"], extra)}
                    log.info("Title and description localised: %s", ", ".join(localizations) or "none")
                except Cancelled:
                    raise
                except Exception as e:  # a nicety: never fail the video over it
                    log.warning("Localising the title and description failed: %s", redact(e))
            # Record actual visual coverage, not just the number of requested scenes. This makes
            # fallback-heavy renders visible in the library and gives the next quality pass real data.
            visual_total = len(media)
            # A file existing is not enough: downloaded stock must have a meaningful metadata
            # match to its own scene. Prompt-generated AI images are scene-directed by construction;
            # generic library art and gradients never count as evidence.
            scene_matches = 0
            visual_audit = []
            for shot, item in zip(shots, media):
                scene = script.scenes[shot.index] if shot.index < len(script.scenes) else None
                scene_query = scene.search_query if scene else script.topic
                metadata = item.text or item.extra.get("description") or item.extra.get("categories")
                match_score = relevance(item, scene_query) if metadata else 0.0
                file_ok = item.path is not None and item.path.is_file()
                is_ai_visual = item.source in AI_SOURCES and file_ok
                is_matched_stock = (
                    item.source not in ("generated", "library") and item.source not in AI_SOURCES
                    and file_ok and bool(metadata) and match_score >= 0.58
                )
                matched = bool(is_ai_visual or is_matched_stock)
                if matched:
                    scene_matches += 1
                visual_audit.append({
                    "shot": len(visual_audit) + 1,
                    "scene": shot.index + 1,
                    "start": round(shot.start, 2),
                    "end": round(shot.end, 2),
                    "query": scene_query,
                    "source": item.source,
                    "asset_id": item.id,
                    "metadata_available": bool(metadata),
                    "relevance_score": round(match_score, 3),
                    "file_available": file_ok,
                    "scene_match_approved": matched,
                    "review_note": (
                        "AI-generated scene visual" if is_ai_visual else
                        "Metadata match accepted" if is_matched_stock else
                        "Fallback or weakly evidenced match; manual review recommended"
                    ),
                })
            visual_real = scene_matches
            visual_coverage = round(100 * visual_real / visual_total) if visual_total else 0
            if visual_coverage < 80:
                log.warning("Visual coverage is only %d%% (%d/%d shots have real media)",
                            visual_coverage, visual_real, visual_total)
            else:
                log.info("Visual coverage: %d%% (%d/%d shots have real media)",
                         visual_coverage, visual_real, visual_total)
            target_delta = abs(info["duration"] - s.target_seconds) / max(1, s.target_seconds)
            duration_ok = target_delta <= 0.20
            visuals_ok = visual_coverage >= 70
            captions_ok = bool(chunks) and all(
                chunk.end > chunk.start
                and chunk.start >= 0
                and chunk.end <= info["duration"] + 0.25
                and bool(chunk.text.strip())
                for chunk in chunks
            ) and all(
                a.start <= b.start and a.end <= b.start + 0.08
                for a, b in zip(chunks, chunks[1:])
            )
            publish_ready = duration_ok and visuals_ok and captions_ok
            quality_report = {
                "target_seconds": s.target_seconds,
                "actual_seconds": round(info["duration"], 2),
                "duration_deviation_percent": round(target_delta * 100, 1),
                "duration_ok": duration_ok,
                "visual_coverage_percent": visual_coverage,
                "visual_coverage_ok": visuals_ok,
                "visual_audit": visual_audit,
                "captions_ok": captions_ok,
                "publish_ready": publish_ready,
                "reasons": [],
            }
            if not duration_ok:
                quality_report["reasons"].append(
                    f"Runtime differs from requested {s.target_seconds}s by more than 20%; revise narration."
                )
                log.warning("Quality gate: rendered runtime %.2fs is too far from target %ss",
                            info["duration"], s.target_seconds)
            if not visuals_ok:
                quality_report["reasons"].append(
                    "Fewer than 70% of shots use scene assets; generic artwork/gradients do not count as visual evidence."
                )
                log.warning("Quality gate: visual coverage is only %d%%", visual_coverage)
            if not captions_ok:
                quality_report["reasons"].append(
                    "Caption timing is empty, reversed, or out of sequence; inspect narration alignment."
                )
                log.warning("Quality gate: caption timing validation failed")
            (folder / "quality_report.json").write_text(
                json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            meta = {
                "title": title, "description": description, "tags": script.tags,
                "category_id": script.category_id, "language": s.language,
                "topic": script.topic, "source": pick.source, "style": script.style,
                "duration": round(info["duration"], 2), "resolution": f"{s.width}x{s.height}",
                "llm": writer, "score": script.score, "rubric": script.rubric, "review": script.review,
                "search_keywords": pick.keywords, "localizations": localizations, "cover_at": round(cover_at, 2),
                "cast": script.cast,
                "voice": f"{speech.engine}:{speech.voice}", "timing": speech.timing,
                "visual_coverage_percent": visual_coverage,
                "visual_matched_shots": visual_real, "visual_real_shots": visual_real,
                "visual_total_shots": visual_total,
                "quality": quality_report,
                "visuals": [{"source": m.source, "id": m.id, "kind": m.kind, "query": m.query}
                            for m in media + ([background] if split else [])],
                "parent_id": parent_id, "created_at": now_iso(),
            }
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

            rid = self.history.add_video(
                subject=pick.subject, source=pick.source, topic=script.topic, style=script.style,
                language=s.language, title=title, description=description, tags=script.tags,
                folder=str(folder), video_path=str(video), duration=info["duration"],
                llm=meta["llm"], voice=meta["voice"],
                # A failed publishing gate is a review state only when publication was requested.
                # A local/offline render is still a completed render; its quality report retains
                # the gate failure and automatic publishing remains blocked.
                status="needs-review" if upload and not publish_ready else "rendered",
                score=script.score, parent_id=parent_id)
            state["record_id"] = rid
            if not media_map:
                self.history.mark_media([m.key for m in media if m.source not in REUSABLE_SOURCES])
            if pick.key and not parent_id:
                self.history.mark_topic(pick.key)
            if pick.idea_id and not parent_id:
                self.history.link_idea(pick.idea_id, rid)

            took = time.time() - t0
            log.info("Rendered %s (%.1fs video) in %.0fs -> %s", title, info["duration"], took, video)
            result = Result(True, title, folder, video, "rendered", seconds=took, record_id=rid,
                            score=script.score, language=s.language)
            if upload and publish_ready:
                self._step("upload")
                self.upload_record(rid, result)
            elif upload:
                result.status = "needs-review"
                result.reason = "; ".join(quality_report["reasons"])
                log.error("Publishing blocked by quality gate: %s", result.reason)
                notify.send(s, "failed", title=title, detail="Publishing blocked for review: " + result.reason)
            else:
                notify.send(s, "rendered", title=title, detail=str(video))

            for lang in [x for x in dict.fromkeys(languages or []) if x != s.language]:
                self._step("translate", lang)
                # A translated copy must not bypass a failed quality gate on the source video.
                result.variants.append(self._translation(
                    script, pick, writer, lang, s, by_scene, credits, rid,
                    upload and publish_ready,
                ))
            self._step("done", title)
            return result
        finally:
            if not os.getenv("KEEP_WORK"):
                shutil.rmtree(work, ignore_errors=True)

    def _translation(self, script: Script, pick: Topic, writer: str, lang: str, s: Settings,
                     by_scene: dict[tuple[int, int], MediaItem], credits: set[str], parent_id: int,
                     upload: bool) -> Result:
        t0 = time.time()
        try:
            if self.llm is None:
                raise RuntimeError("translations need an LLM (not available in demo mode)")
            translated = translate_script(self.llm, script, lang)
            translated.score, translated.review, translated.rubric = script.score, script.review, script.rubric
            log.info("Translated to %s: %s", lang, translated.title)
            self._quiet = True  # the copy's own stages are one "translate" step for the progress bar
            try:
                return self.produce(translated, pick, writer, upload=upload, t0=t0, media_map=by_scene,
                                    credits=credits, parent_id=parent_id,
                                    settings=replace(s, language=lang, tts_voice="", tts_voice_b=""))
            finally:
                self._quiet = False
        except (youtube.NotAuthorized, Cancelled):
            raise
        except Exception as e:  # including LLMConfigError: the original video is already made
            log.error("Translation to %s failed: %s", lang, redact(e))
            return Result(False, status="failed", error=redact(e), language=lang, seconds=time.time() - t0)

    # ------------------------------------------------------------------ uploading
    @property
    def _refused_on(self) -> str:
        """The quota day on which YouTube itself refused an upload. Kept in the history database, so every
        Studio job, a restarted autopilot and the MCP server all know it."""
        return self.history.get_meta("youtube_refused_on") or ""

    @_refused_on.setter
    def _refused_on(self, day: str) -> None:
        self.history.set_meta("youtube_refused_on", day or None)

    def quota_left(self) -> int:
        day = quota_day_start()
        if self._refused_on == day:  # YouTube said no today: nothing more goes up until the reset
            return 0
        return self.s.daily_upload_limit - self.history.uploads_since(day)

    def upload_record(self, rid: int, result: Result | None = None) -> str:
        """Publish a rendered video to every PUBLISH_TO destination it hasn't reached yet: YouTube (uploaded,
        or scheduled into the next free slot) and the other networks (see crosspost). Calling it again only
        retries what didn't go out. Returns the video's new status."""
        rec = self.history.get(rid)
        if not rec or not rec.get("video_path") or not Path(rec["video_path"]).exists():
            log.error("Video #%s has no file to upload", rid)
            return "missing"
        if not self.s.upload:
            message = "Publishing is disabled (UPLOAD=false). Enable it explicitly in .env before uploading."
            log.warning("%s", message)
            if result:
                result.status, result.error = "disabled", message
            return "disabled"
        to = list(dict.fromkeys(self.s.publish_to or ["youtube"]))
        folder = Path(rec["folder"])
        meta_file = folder / "metadata.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        posted = dict(meta.get("published") or {})
        on_youtube = "youtube" in to and not rec.get("youtube_id")
        others = [p for p in to if p != "youtube" and (posted.get(p) or {}).get("status") not in crosspost.DONE]
        if not on_youtube and not others:
            log.info("Video #%d is already on %s", rid, ", ".join(to))
            return rec["status"]
        if self.quota_left() <= 0:
            log.warning("Daily upload limit (%d) reached — video #%d queued for tomorrow", self.s.daily_upload_limit, rid)
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="daily upload limit reached")
            return "queued"
        status = rec["status"]
        when = _future(rec.get("publish_at"))
        if on_youtube:
            # A creator-selected time takes precedence over configured default slots.
            when = _future(rec.get("publish_at")) or youtube.next_publish_slot(self.s, self.history.scheduled_times())
            status = self._to_youtube(rid, rec, meta, folder, when, result)
            if status not in ("uploaded", "scheduled"):
                when = None
        elif "youtube" not in to and when is None and not posted:
            when = youtube.next_publish_slot(self.s, self.history.scheduled_times())  # the slot is for the networks
        if others:
            results = crosspost.publish(self.s, Path(rec["video_path"]), meta, others, when)
            posted.update({p: r.to_dict() for p, r in results.items()})
            meta["published"] = posted
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            out = [r for r in results.values() if r.status in crosspost.DONE]
            failed = [r for r in results.values() if r.status not in crosspost.DONE]
            for r in out:
                notify.send(self.s, "uploaded", title=f"{rec['title']} on {r.platform}", url=r.url, detail=r.status)
            for r in failed:
                notify.send(self.s, "failed", title=f"{rec['title']} on {r.platform}", detail=r.error)
            if "youtube" not in to:  # without YouTube, the video's status is where the networks stand
                ok = bool(out) or any((posted.get(p) or {}).get("status") in crosspost.DONE for p in to)
                status = ("scheduled" if when and out and all(r.status == "scheduled" for r in out) else
                          "uploaded") if ok else "failed"
                self.history.update_video(
                    rid, status=status, uploaded_at=now_iso() if ok else rec.get("uploaded_at"),
                    publish_at=youtube.to_rfc3339(when) if when and ok else rec.get("publish_at"),
                    error=None if not failed else "; ".join(f"{r.platform}: {r.error}" for r in failed)[:500])
            if result:
                result.posts = {p: posted[p] for p in others}
                if "youtube" not in to:
                    result.status = status
        # The MP4 can go once every destination has it (KEEP_VIDEOS=false); otherwise a retry needs it.
        youtube_done = "youtube" not in to or status in ("uploaded", "scheduled")
        everywhere = youtube_done and all((posted.get(p) or {}).get("status") in crosspost.DONE
                                          for p in to if p != "youtube")
        if everywhere and not self.s.keep_videos:
            Path(rec["video_path"]).unlink(missing_ok=True)
            self.history.update_video(rid, video_path=None)
        return status

    def _to_youtube(self, rid: int, rec: dict, meta: dict, folder: Path, publish, result: Result | None) -> str:
        body = youtube.build_body(self.s, rec["title"], rec["description"] or "",
                                  json.loads(rec["tags"] or "[]"), meta.get("category_id", "27"),
                                  meta.get("language", self.s.language), publish,
                                  localizations=meta.get("localizations"))
        try:
            resp = youtube.upload(self.s, Path(rec["video_path"]), body)
        except youtube.QuotaExceeded:
            log.warning("YouTube refused the upload (quota or upload limit) — video #%d queued until midnight Pacific",
                        rid)
            self._refused_on = quota_day_start()
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="YouTube's upload limit reached")
            return "queued"
        except youtube.NotAuthorized:
            raise
        except Exception as e:
            log.error("Upload of #%d failed: %s", rid, e)
            self.history.update_video(rid, status="failed", error=str(e)[:500])
            if result:
                result.status, result.error = "upload-failed", str(e)
            notify.send(self.s, "failed", title=rec["title"], detail=f"upload failed: {redact(e)}")
            return "failed"
        vid = resp.get("id")
        if vid and self.s.upload_captions:
            lang = meta.get("language", self.s.language)
            youtube.upload_captions(self.s, vid, folder / "captions.srt", lang,
                                    LANGUAGES.get(lang.split("-")[0], lang))
        status = "scheduled" if publish else "uploaded"
        publish_iso = youtube.to_rfc3339(publish) if publish else None
        self.history.update_video(rid, status=status, youtube_id=vid, uploaded_at=now_iso(), publish_at=publish_iso,
                                  error=None)
        when = f", goes public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else ""
        log.info("%s: https://youtube.com/shorts/%s%s", status.capitalize(), vid, when)
        notify.send(self.s, status, title=rec["title"], url=f"https://youtube.com/shorts/{vid}",
                    detail=f"public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else "")
        if result:
            result.status, result.youtube_id, result.publish_at = status, vid, publish_iso
        return status

    def flush_queue(self, include_rendered: bool = False) -> int:
        statuses = ("queued", "rendered") if include_rendered else ("queued",)
        done = 0
        for rec in self.history.pending_uploads(statuses):
            if self.quota_left() <= 0:
                break
            if self.upload_record(rec["id"]) in ("uploaded", "scheduled"):
                done += 1
        return done

    def delete(self, rid: int) -> bool:
        """Remove a video from the library and delete its folder (never touches YouTube)."""
        rec = self.history.get(rid)
        if not rec:
            return False
        folder = Path(rec["folder"]) if rec.get("folder") else None
        if folder and folder.is_dir() and folder.resolve().is_relative_to(self.s.out_path.resolve()):
            shutil.rmtree(folder, ignore_errors=True)
        self.history.delete_video(rid)
        return True

    # ------------------------------------------------------------------ autopilot
    def autopilot(self, *, count: int | None = None, once: bool = False, upload: bool | None = None,
                  topic: str | None = None, source: str | None = None) -> int:
        s = self.s
        upload = s.upload if upload is None else upload
        produced, batch, empty_batches = 0, 0, 0
        mode = "upload" if upload else "no-upload"
        _ = self.llm  # fail fast on LLM configuration problems
        log.info("Autopilot started (%s, batch %d, workers %d%s)", mode, s.batch_size, s.workers,
                 f", target {count}" if count else "")
        try:
            while True:
                batch += 1
                if s.learn_from_stats and time.time() - self._stats_synced > 12 * 3600:
                    analytics.sync_stats(s, self.history)
                    self._stats_synced = time.time()
                if upload:
                    self.flush_queue()
                    if self.quota_left() <= 0 and self.history.pending_uploads():
                        wait = seconds_until_quota_reset()
                        log.info("Upload limit reached and videos are queued; sleeping %.1fh until quota resets",
                                 wait / 3600)
                        time.sleep(wait)
                        continue
                n = s.batch_size if count is None else min(s.batch_size, count - produced)
                workers = min(s.workers, n)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    # Parallel videos each get their own Studio: progress, timing and cancel state are per video.
                    makers = [self] if workers == 1 else [Studio(s) for _ in range(n)]
                    futs = [ex.submit(makers[i % len(makers)].make, topic, source=source, upload=upload)
                            for i in range(n)]
                    results = [f.result() for f in futs]
                ok = sum(r.ok for r in results)
                produced += ok
                log.info("Batch %d: %d/%d succeeded (total %d)", batch, ok, n, produced)
                empty_batches = empty_batches + 1 if ok == 0 else 0
                if once or (count is not None and produced >= count):
                    break
                if empty_batches >= 3:
                    log.error("Three batches in a row produced nothing — stopping. Check the log above.")
                    break
                if s.batch_delay:
                    log.info("Next batch in %ds", s.batch_delay)
                    time.sleep(s.batch_delay)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        return produced
