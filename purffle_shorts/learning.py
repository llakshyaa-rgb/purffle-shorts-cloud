"""Persistent, evidence-backed creator notes for Purffle's next video."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import analytics
from .config import Settings
from .history import History

META_KEY = "creator_learning_v1"
MANUAL_KEY = "creator_manual_notes_v1"


def read_learning(history: History) -> dict:
    try:
        data = json.loads(history.get_meta(META_KEY) or "{}")
        if not isinstance(data, dict):
            data = {}
    except (TypeError, ValueError):
        data = {}
    manual = history.get_meta(MANUAL_KEY) or ""
    data["manual_notes"] = manual
    data["prompt_notes"] = "\n\n".join(x for x in (str(data.get("notes") or ""), manual.strip()) if x)
    return data


def save_manual_notes(history: History, notes: str) -> dict:
    history.set_meta(MANUAL_KEY, (notes or "").strip()[:5000])
    return read_learning(history)


def refresh_learning(settings: Settings, history: History, *, sync: bool = True) -> dict:
    """Refresh channel stats and persist a compact, inspectable summary for future scripts."""
    updated = analytics.sync_stats(settings, history) if sync else 0
    rows = [r for r in history.recent(1000) if r.get("youtube_id")]
    with_stats = [r for r in rows if r.get("views") is not None]
    with_retention = [r for r in with_stats if r.get("average_view_percentage") is not None]
    mature = history.results()
    evidence = analytics.insights(history)
    calibration = analytics.calibration(history, settings.timezone)
    lines = [
        f"Last refreshed: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Channel evidence: {len(rows)} uploaded videos tracked; {len(with_stats)} have view counts; "
        f"{len(with_retention)} have average-view percentage; {len(mature)} videos have mature (48+ hour) stats.",
    ]
    if evidence:
        lines.append(evidence)
    elif len(with_stats) < 3:
        lines.append("Early-learning stage: not enough mature performance data yet. Keep topics varied and gather more uploads; do not overfit to one video's views.")
    else:
        lines.append("Performance comparisons are available, but there are not yet enough mature videos for reliable pattern notes. Treat early signals as tentative.")
    style_rows = calibration.get("styles") or []
    if style_rows:
        ranked = sorted(style_rows, key=lambda r: (r.get("median_views") or 0), reverse=True)
        lines.append("Format signal (median views, descriptive only): " + "; ".join(
            f"{r['style']} ({r['videos']} videos, median {r['median_views']:,} views)"
            for r in ranked[:4]))
    topic_rows = calibration.get("topics") or []
    if topic_rows:
        lines.append("Repeated-topic signal: " + "; ".join(
            f"{r['topic']} ({r['videos']} videos, median {r['median_views']:,} views)"
            for r in topic_rows[:4]))
    lines.append("Next-video rules: choose a fresh topic automatically from configured niches/topic sources; avoid recent titles and duplicate angles; use repeated performance patterns as hypotheses, not guarantees; never sacrifice factual accuracy or story clarity for views.")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats_updated": updated,
        "tracked_videos": len(rows),
        "videos_with_stats": len(with_stats),
        "videos_with_retention": len(with_retention),
        "mature_videos": len(mature),
        "notes": "\n\n".join(lines),
        "calibration": calibration,
    }
    history.set_meta(META_KEY, json.dumps(payload, ensure_ascii=False))
    return read_learning(history)
