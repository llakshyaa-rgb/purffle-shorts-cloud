"""Persistent, reusable library for locally drawn scene artwork.

Assets are deterministic and stored outside per-video work folders, so the same visual can be
reused by later videos without another API call. The hidden cache folder is deliberately not
treated as user-provided footage.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .visual_library import character_kind, make_visual, visual_type

_LOCK = threading.RLock()
_INDEX = "index.json"
MAX_ASSETS = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_index(root: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads((root / _INDEX).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_index(root: Path, index: dict[str, dict[str, Any]]) -> None:
    tmp = root / (_INDEX + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(root / _INDEX)


def asset_key(query: str, prompt: str, width: int, height: int, variation: int,
              mode: str, style_key: str) -> str:
    if mode == "character":
        # One canonical asset per named character and canvas size, independent of scene wording.
        parts = ["character", style_key.strip().casefold(), int(width), int(height)]
    else:
        parts = [query.strip().casefold(), prompt.strip().casefold(), int(width), int(height),
                 int(variation), mode, style_key.strip().casefold()]
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def get_or_create_asset(
    root: str | Path,
    query: str,
    prompt: str,
    width: int,
    height: int,
    variation: int = 0,
    mode: str = "",
    style_key: str = "",
) -> dict[str, Any]:
    """Return a persistent asset and copy it to the requested per-video output separately."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = asset_key(query, prompt, width, height, variation, mode, style_key)
    filename = key + ".png"
    path = root / filename
    with _LOCK:
        index = _read_index(root)
        record = index.get(key, {})
        if not path.is_file():
            kind = make_visual(
                query, prompt, path, width, height, variation=variation,
                mode=mode or None, style_key=style_key,
            )
            record = {
                "id": key,
                "file": filename,
                "query": query[:240],
                "prompt": prompt[:800],
                "visual_type": kind,
                "character": character_kind(query + " " + prompt),
                "style_key": style_key[:160],
                "width": int(width),
                "height": int(height),
                "created_at": _now(),
                "uses": 0,
            }
        record["uses"] = int(record.get("uses") or 0) + 1
        record["last_used_at"] = _now()
        index[key] = record
        # Keep the persistent library bounded on low-cost devices and small cloud disks.
        if len(index) > MAX_ASSETS:
            oldest = sorted(
                (r for r in index.values() if r.get("id") != key),
                key=lambda r: r.get("last_used_at") or r.get("created_at") or "",
            )[:len(index) - MAX_ASSETS]
            for old in oldest:
                old_key = str(old.get("id") or "")
                old_file = root / str(old.get("file") or "")
                try:
                    if old_file.parent.resolve() == root.resolve() and old_file.suffix == ".png":
                        old_file.unlink(missing_ok=True)
                except OSError:
                    pass
                index.pop(old_key, None)
        _write_index(root, index)
    return {**record, "path": path}


def list_assets(root: str | Path, limit: int = 120) -> list[dict[str, Any]]:
    root = Path(root)
    with _LOCK:
        index = _read_index(root) if root.is_dir() else {}
    rows = []
    for key, record in index.items():
        filename = str(record.get("file") or "")
        path = root / filename
        if key != record.get("id") or not path.is_file() or path.parent.resolve() != root.resolve():
            continue
        rows.append({**record, "id": key})
    rows.sort(key=lambda r: r.get("last_used_at") or r.get("created_at") or "", reverse=True)
    return rows[:max(1, min(int(limit), MAX_ASSETS))]


def asset_path(root: str | Path, key: str) -> Path | None:
    if len(key) != 24 or any(ch not in "0123456789abcdef" for ch in key):
        return None
    root = Path(root)
    with _LOCK:
        record = _read_index(root).get(key)
    if not record or record.get("file") != key + ".png":
        return None
    path = root / (key + ".png")
    return path if path.is_file() and path.parent.resolve() == root.resolve() else None
