"""Background scene-level creative direction for Purffle Shorts.

The director returns actionable shot instructions plus concise, user-facing rationales.
It never stores hidden chain-of-thought. If the configured LLM is unavailable, a local
rule-based plan keeps production moving without another dependency or paid service.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .script import Script

log = logging.getLogger("purffle")

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "purpose": {"type": "string"},
                    "visual_mode": {"type": "string", "enum": ["illustration", "diagram", "footage", "photo", "symbolic"]},
                    "visual_prompt": {"type": "string"},
                    "camera": {"type": "string", "enum": ["push_in", "pull_out", "pan_left", "pan_right", "static", "shake"]},
                    "transition": {"type": "string", "enum": ["cut", "fade", "dissolve", "whip", "match_cut"]},
                    "sound": {"type": "string", "enum": ["none", "whoosh", "pop", "hit"]},
                    "reason": {"type": "string"},
                },
                "required": ["scene", "purpose", "visual_mode", "visual_prompt", "camera", "transition", "sound", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenes"],
    "additionalProperties": False,
}

CAMERAS = ("push_in", "pull_out", "pan_left", "pan_right", "static", "shake")
MODES = ("illustration", "diagram", "footage", "photo", "symbolic")
TRANSITIONS = ("cut", "fade", "dissolve", "whip", "match_cut")
SOUNDS = ("none", "whoosh", "pop", "hit")


def _fallback(scene: Any, index: int, style: str) -> dict[str, Any]:
    text = f"{getattr(scene, 'narration', '')} {getattr(scene, 'search_query', '')} {getattr(scene, 'image_prompt', '')}".lower()
    diagram_words = ("how", "why", "process", "system", "flow", "cause", "effect", "brain", "orbit", "energy", "moves", "works", "inside", "mechanism", "steps")
    story_words = ("door", "room", "shadow", "dark", "heard", "knock", "footsteps", "ran", "screamed", "blood", "monster", "stranger", "alone", "silhouette")
    if any(w in text for w in diagram_words) and style not in ("story", "reddit", "chat"):
        mode = "diagram"
    elif any(w in text for w in story_words) or style in ("story", "reddit", "chat"):
        mode = "illustration"
    else:
        mode = "footage"
    camera = ("push_in", "pan_left", "push_in", "pan_right", "pull_out", "static")[index % 6]
    if any(w in text for w in ("reveal", "realize", "suddenly", "impossible", "finally", "truth")):
        camera, sound, transition = "push_in", "hit", "cut"
    elif index == 0:
        sound, transition = "hit", "cut"
    else:
        sound, transition = ("whoosh" if index % 2 else "none"), ("match_cut" if index % 3 == 0 else "cut")
    query = getattr(scene, "search_query", "") or getattr(scene, "narration", "")
    prompt = getattr(scene, "image_prompt", "") or query
    mode_hint = {
        "illustration": "original cinematic illustrated scene, clear readable silhouette and specific visible action",
        "diagram": "clean custom explanatory diagram, labeled only when genuinely useful, simple visual cause-and-effect",
        "footage": "documentary-style real-world footage, one clear subject and visible action",
        "photo": "cinematic editorial photograph, realistic lighting and a single focal subject",
        "symbolic": "visual metaphor with one unmistakable central symbol, cinematic composition",
    }[mode]
    return {
        "scene": index + 1,
        "purpose": "hook" if index == 0 else ("payoff" if any(w in text for w in ("therefore", "because", "finally", "the truth", "so that")) else "evidence"),
        "visual_mode": mode,
        "visual_prompt": f"{prompt}. {mode_hint}. Portrait 9:16 framing; no decorative text, no collage, no unrelated stock imagery.",
        "camera": camera,
        "transition": transition,
        "sound": sound,
        "reason": f"Use {mode} to make the narrated beat immediately legible; {camera.replace('_', ' ')} keeps the shot alive without distracting from the words.",
    }


def direct(script: Script, llm: Any = None) -> list[dict[str, Any]]:
    """Plan each scene; one structured LLM call, with a deterministic local fallback."""
    fallback = [_fallback(scene, i, script.style) for i, scene in enumerate(script.scenes)]
    if llm is None or not script.scenes:
        return fallback
    payload = {
        "topic": script.topic,
        "title": script.title,
        "style": script.style,
        "scenes": [
            {"scene": i + 1, "narration": sc.narration, "search_query": sc.search_query,
             "existing_visual_prompt": sc.image_prompt}
            for i, sc in enumerate(script.scenes)
        ],
    }
    system = (
        "You are the scene-level creative director for a vertical short-form video. "
        "Plan every scene in the background. Return practical instructions the renderer can use, not an essay. "
        "Choose a visual mode scene-by-scene: diagrams for causal/mechanical explanations; original illustrations "
        "for fictional/horror beats; footage/photos for real-world evidence; symbolic visuals only when clearer. "
        "Avoid repeating the same visual treatment for every scene. The prompt must show the actual narrated subject/action, "
        "setting, composition and lighting, not merely restate the topic. Do not request visible words except labels essential "
        "to a diagram. Camera moves must be subtle and mobile-safe. Transitions should serve continuity, not decorate every cut. "
        "Sound effects are sparse accents; choose none often and never drown out narration. Rationale is a short user-facing "
        "production note (one sentence), not private chain-of-thought. Keep all facts in the provided script; do not add claims."
    )
    try:
        result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), schema=DIRECTOR_SCHEMA,
                                   max_tokens=min(4500, 500 + len(script.scenes) * 150), temperature=0.35)
        planned = result.get("scenes", [])
        by_scene = {int(row.get("scene", 0)): row for row in planned if isinstance(row, dict)}
        merged = []
        for i, base in enumerate(fallback, 1):
            row = by_scene.get(i, {})
            clean = dict(base)
            for key in ("purpose", "visual_prompt", "reason"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    clean[key] = value.strip()[:1800 if key == "visual_prompt" else 240]
            for key, allowed in (("visual_mode", MODES), ("camera", CAMERAS),
                                 ("transition", TRANSITIONS), ("sound", SOUNDS)):
                if row.get(key) in allowed:
                    clean[key] = row[key]
            clean["scene"] = i
            merged.append(clean)
        return merged
    except Exception as exc:
        log.warning("Creative Director unavailable; using local direction: %s", str(exc)[:180])
        return fallback


def apply_to_script(script: Script, plan: list[dict[str, Any]]) -> None:
    """Apply the director's visual choices to the actual media prompts."""
    for scene, beat in zip(script.scenes, plan):
        prompt = str(beat.get("visual_prompt", "")).strip()
        if prompt:
            scene.image_prompt = prompt


def notes_markdown(script: Script, plan: list[dict[str, Any]]) -> str:
    lines = [f"# Director's Notes — {script.title}", "", "Background production plan. These are concise decisions, not private reasoning.", ""]
    for beat in plan:
        i = beat["scene"]
        narration = script.scenes[i - 1].narration if 0 < i <= len(script.scenes) else ""
        lines += [
            f"## Scene {i}: {beat.get('purpose', 'beat').title()}",
            f"- **Narration:** {narration}",
            f"- **Visual:** {beat.get('visual_mode', 'illustration').title()} — {beat.get('visual_prompt', '')}",
            f"- **Camera:** {beat.get('camera', 'static').replace('_', ' ').title()}",
            f"- **Transition:** {beat.get('transition', 'cut').replace('_', ' ').title()}",
            f"- **Sound accent:** {beat.get('sound', 'none').title()}",
            f"- **Why:** {beat.get('reason', '')}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
