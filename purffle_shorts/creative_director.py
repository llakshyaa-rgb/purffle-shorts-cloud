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
from .fantasy_video_library import FANTASY_VISUAL_RULES

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
                    "visual_mode": {"type": "string", "enum": ["illustration", "diagram", "footage", "photo", "symbolic", "gradient"]},
                    "visual_prompt": {"type": "string"},
                    "camera": {"type": "string", "enum": ["push_in", "pull_out", "pan_left", "pan_right", "static", "shake"]},
                    "transition": {"type": "string", "enum": ["cut", "fade", "dissolve", "whip", "match_cut"]},
                    "sound": {"type": "string", "enum": ["none", "whoosh", "pop", "hit"]},
                    "reason": {"type": "string"},
                    "character": {"type": "string", "enum": ["none", "purffle_narrator", "purffle_detective", "purffle_scientist", "purffle_explorer", "purffle_mystery"]},
                },
                "required": ["scene", "purpose", "visual_mode", "visual_prompt", "camera", "transition", "sound", "reason", "character"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenes"],
    "additionalProperties": False,
}

CAMERAS = ("push_in", "pull_out", "pan_left", "pan_right", "static", "shake")
MODES = ("illustration", "diagram", "footage", "photo", "symbolic", "gradient")
TRANSITIONS = ("cut", "fade", "dissolve", "whip", "match_cut")
SOUNDS = ("none", "whoosh", "pop", "hit")
CHARACTERS = ("none", "purffle_narrator", "purffle_detective", "purffle_scientist", "purffle_explorer", "purffle_mystery")


def _character_for(text: str, style: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & {"detective", "investigator", "sleuth"}:
        return "purffle_detective"
    if words & {"scientist", "researcher", "laboratory"}:
        return "purffle_scientist"
    if words & {"explorer", "astronaut", "adventurer"}:
        return "purffle_explorer"
    if words & {"ghost", "monster", "stranger", "shadow", "mysterious", "creature"}:
        return "purffle_mystery"
    if style in ("story", "reddit") and words & {"narrator", "hooded", "purffle"}:
        return "purffle_narrator"
    return "none"


def _needs_gradient(narration: str, query: str) -> bool:
    """Use designed gradients for connective narration, never as evidence for concrete claims."""
    line = (narration or "").strip().lower()
    q = (query or "").strip().lower()
    bridges = (
        "which means", "the real question", "here is the catch", "here's the catch",
        "that is why", "that's why", "in other words", "the answer is",
        "so what does that mean", "the takeaway is", "what this tells us",
    )
    if any(phrase in line for phrase in bridges):
        return True
    generic_query = not q or q in {"abstract", "transition", "background", "gradient"}
    concrete = re.search(
        r"\b(octopus|planet|saturn|earth|moon|heart|brain|person|human|animal|"
        r"scientist|experiment|study|researcher|city|ocean|volcano|door|car|"
        r"machine|cell|blood|water|fire|ice|camera|phone|satellite|evidence)\b", line
    )
    return generic_query and len(line.split()) <= 6 and not concrete


def _fallback(scene: Any, index: int, style: str) -> dict[str, Any]:
    narration = str(getattr(scene, "narration", "") or "")
    query = str(getattr(scene, "search_query", "") or "")
    text = f"{narration} {query} {getattr(scene, 'image_prompt', '')}".lower()
    # Abstract connective beats should use designed motion graphics, not irrelevant stock.
    abstract_phrases = ("which means", "the real question", "here is the catch", "here's the catch",
                        "that is why", "that's why", "in other words", "the answer is",
                        "so what does that mean")
    generic_query = not query or query.strip().lower() in {"abstract", "transition", "background", "gradient"}
    if any(phrase in narration.lower() for phrase in abstract_phrases) or (generic_query and len(narration.split()) <= 8):
        mode = "gradient"
    # Reserve diagram treatment for an actual mechanism, not generic words like "why" or "brain".
    # Broad triggers made unrelated factual scenes collapse into the same flat infographic style.
    diagram_words = ("process", "flowchart", "cause and effect", "mechanism", "cross-section", "cutaway", "step by step", "orbit around", "gravity pulls", "energy transfer")
    story_words = ("door", "room", "shadow", "dark", "heard", "knock", "footsteps", "ran", "screamed", "blood", "monster", "stranger", "alone", "silhouette")
    if style == "fantasy":
        # Fantasy role-play should look like a cinematic scene, not our offline vector library.
        mode = "footage"
    elif any(w in text for w in diagram_words) and style not in ("story", "reddit", "chat", "fantasy"):
        mode = "diagram"
    elif style in ("story", "reddit", "chat") or (
        style not in ("facts", "explainer", "news", "quiz", "myth")
        and any(w in text for w in story_words)
    ):
        mode = "illustration"
    else:
        mode = "footage"
    # A deterministic visual rhythm: factual shorts should not default every beat to a diagram.
    # Use footage for concrete subjects and reserve diagrams for explicit mechanisms only.
    if mode == "diagram" and index % 3 == 2:
        mode = "footage"
    # Re-apply this after the generic footage/illustration selection above: abstract beats
    # must not accidentally be overwritten into stock-footage mode.
    if any(phrase in narration.lower() for phrase in abstract_phrases) or (
        generic_query and len(narration.split()) <= 8
    ):
        mode = "gradient"
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
        "gradient": "intentional animated gradient background, deep cinematic color, restrained slow light movement, clean negative space for captions; no people, no stock photo, no icons",
        "illustration": "original cinematic illustrated scene, clear readable silhouette and specific visible action",
        "diagram": "clean custom explanatory diagram, labeled only when genuinely useful, simple visual cause-and-effect",
        "footage": ("cinematic live-action fantasy footage or high-end realistic fantasy-game cinematic, " 
                    "real actors or photoreal CGI creatures, one clear subject and visible action, " 
                    "dramatic natural camera movement, detailed environments; no cartoon, no flat drawing, " 
                    "no vector art, no icons, no subtitles, no UI, no logos, no watermarks") if style == "fantasy" else "documentary-style real-world footage, one clear subject and visible action",
        "photo": "cinematic editorial photograph, realistic lighting and a single focal subject",
        "symbolic": "visual metaphor with one unmistakable central symbol, cinematic composition",
    }[mode]
    return {
        "scene": index + 1,
        "purpose": "hook" if index == 0 else ("payoff" if any(w in text for w in ("therefore", "because", "finally", "the truth", "so that")) else "evidence"),
        "visual_mode": mode,
        "visual_prompt": (f"FANTASY CINEMATIC FOOTAGE: {prompt}. {mode_hint}. " if style == "fantasy" else f"{prompt}. {mode_hint}. ") + "Portrait 9:16 framing; no decorative text, no collage, no unrelated stock imagery.",
        "camera": camera,
        "transition": transition,
        "sound": sound,
        "reason": f"Use {mode} to make the narrated beat immediately legible; {camera.replace('_', ' ')} keeps the shot alive without distracting from the words.",
        "character": _character_for(text, style),
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
        "Return valid JSON matching the provided schema. Plan every scene in the background. "
        "Return practical instructions the renderer can use, not an essay. "
        "Choose a visual mode scene-by-scene: use real footage/photos for concrete factual subjects and evidence, diagrams only "
        "for mechanisms that genuinely need explaining, original illustrations for fictional/horror beats, and symbolic visuals only when clearer. "
        "Gradient is an intentional first-class mode, not a failed search fallback: use it for abstract connective lines, transitions, "
        "or narration where no specific image materially helps. Never insert a random human or generic stock photo just to fill the screen. "
        "If a person is not part of the narrated event or evidence, do not show one. A restrained animated gradient with negative space is better "
        "than irrelevant imagery. Use gradients sparingly, about one scene in four at most; concrete factual claims still need relevant visuals. "
        "Avoid repeating the same visual treatment for every scene. Every scene must have a distinct, concrete visual "
        "subject or action tied to that scene's narration; do not use the same planet icon, generic orbit diagram, "
        "floating shapes, or broad topic wallpaper as a substitute for the specific fact being explained. "
        "For a sequence about a physical property, show the property in action or a clear comparison; for named objects, "
        "show that object rather than a generic space/technology background. If using a stock-search query, include the "
        "specific subject plus the visible action or setting, not just the broad topic. When a recurring character is visibly present, "
        "reuse the same allowed character archetype across those scenes; choose none when no recurring character is shown. "
        "Never force a mascot into factual footage or assign a character that contradicts the narration. "
        "The prompt must show the actual narrated subject/action, "
        "setting, composition and lighting, not merely restate the topic. Do not request visible words except labels essential "
        "to a diagram. Camera moves must be subtle and mobile-safe. Transitions should serve continuity, not decorate every cut. "
        "Sound effects are sparse accents; choose none often and never drown out narration. Rationale is a short user-facing "
        "production note (one sentence), not private chain-of-thought. Keep all facts in the provided script; do not add claims."
    )
    if script.style == "fantasy":
        system += "\n\n" + FANTASY_VISUAL_RULES + "\n\nFor every fantasy scene choose visual_mode=footage, start visual_prompt with FANTASY CINEMATIC FOOTAGE:, and describe the precise physical action, shot size, setting and continuity. Never request drawn/vector art.\n\nCHARACTER REFERENCE AND PERFORMANCE RULES: Create original characters, never use a random real person's photograph as the canonical design. Keep the exact same face, age, hair, skin/scales/fur, clothing/armor, palette and signature props across all scenes. Each visual prompt must name which character is visible and preserve their reference traits. For dialogue, identify the current speaker, show only that character speaking, keep the listener's mouth still, and use a listener reaction shot after the spoken line ends. Align cuts to measured TTS duration/word timestamps when available; do not guess timing or begin the next line early. If a character-reference image is available, use it as the image-generation reference; if unavailable, provide a precise textual reference and do not claim an image was generated."
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
            if row.get("character") in CHARACTERS:
                clean["character"] = row["character"]
            # Enforce the abstract-beat rule after LLM merging too: model output must not
            # turn a connective sentence into a random stock photo or human subject.
            source_scene = script.scenes[i - 1]
            if _needs_gradient(source_scene.narration, source_scene.search_query):
                clean["visual_mode"] = "gradient"
                clean["character"] = "none"
                clean["visual_prompt"] = (
                    "Abstract animated gradient transition, deep cinematic color, subtle slow light movement, "
                    "clean negative space for captions; no people, no stock photo, no icons, no collage."
                )
            if script.style == "fantasy":
                clean["visual_mode"] = "footage"
                prompt = clean.get("visual_prompt", "").strip()
                if not prompt.upper().startswith("FANTASY CINEMATIC FOOTAGE:"):
                    prompt = "FANTASY CINEMATIC FOOTAGE: " + prompt
                clean["visual_prompt"] = (prompt + ". Photoreal live-action fantasy film or premium realistic CGI cinematic; exact visible action; consistent character and location; varied film framing; no cartoon, vector art, diagrams, icons, collage, UI, text, logo or watermark.")[:1800]
            clean["scene"] = i
            merged.append(clean)
        return merged
    except Exception as exc:
        log.warning("Creative Director unavailable; using local direction: %s", str(exc)[:180])
        return fallback


def apply_to_script(script: Script, plan: list[dict[str, Any]]) -> None:
    """Apply the director's visual choices to the actual media prompts."""
    fantasy_refs: dict[str, str] = {}
    if script.style == "fantasy":
        # Preserve the generated character bible instead of replacing it with generic archetypes.
        # The exact identity text is reused across every shot of the same named character.
        for name in script.cast[:2]:
            for source_scene in script.scenes:
                source_prompt = str(getattr(source_scene, "image_prompt", "") or "")
                for line in source_prompt.splitlines():
                    marker = "REUSABLE CHARACTER:"
                    if marker not in line.upper():
                        continue
                    identity = line[line.upper().find(marker) + len(marker):].strip()
                    if identity.lower().startswith(name.lower()):
                        identity = identity[len(name):].lstrip(" —-:;")
                        if identity:
                            fantasy_refs[name] = f"{name} — {identity[:600]}"
                            break
                if name in fantasy_refs:
                    break
            if name not in fantasy_refs:
                if len(fantasy_refs) == 0:
                    look = "young adult human mage, copper-red braided hair, amber eyes, weathered teal travel cloak, brass star-shaped clasp, leather spell satchel"
                else:
                    look = "young adult elven swordsman, pale silver hair, dark green eyes, charcoal scale armor, one broken silver ear cuff, narrow black-steel sword"
                fantasy_refs[name] = f"{name} — original fictional character; {look}; cinematic live-action fantasy realism; exact same face, costume, colors and props in every shot"
    for scene, beat in zip(script.scenes, plan):
        prompt = str(beat.get("visual_prompt", "")).strip()
        mode = str(beat.get("visual_mode", "illustration")).strip().lower()
        prefix = {
            "gradient": "ABSTRACT GRADIENT BACKGROUND: ",
            "diagram": "DIAGRAM / EXPLANATORY GRAPHIC: ",
            "illustration": "ORIGINAL DRAWN ILLUSTRATION: ",
            "symbolic": "SYMBOLIC VISUAL METAPHOR: ",
            "footage": "REAL-WORLD DOCUMENTARY FOOTAGE: ",
            "photo": "REALISTIC EDITORIAL PHOTOGRAPH: ",
        }.get(mode, "")
        if script.style == "fantasy" and mode == "footage":
            prefix = "FANTASY CINEMATIC FOOTAGE: "
            prompt = prompt.removeprefix("FANTASY CINEMATIC FOOTAGE: ").strip()
        if prompt:
            if mode == "gradient":
                scene.image_prompt = prefix + prompt
                continue
            character = str(beat.get("character", "none")).strip().lower()
            character_note = ""
            if script.style == "fantasy":
                # Scene speaker A/B maps to cast order. Use the same model-sheet wording
                # throughout the episode to make identity drift less likely.
                speaker_index = 1 if str(getattr(scene, "speaker", "A")).upper() == "B" else 0
                visible_name = script.cast[speaker_index] if speaker_index < len(script.cast) else (script.cast[0] if script.cast else "the protagonist")
                reference = fantasy_refs.get(visible_name, f"{visible_name} — original fictional fantasy character; same recognizable design in every shot")
                # Keep identity and shot instructions on separate lines so the media layer can
                # cache the canonical identity key while generating a different action each time.
                prompt = prompt.removeprefix("FANTASY CINEMATIC FOOTAGE: ").strip()
                character_note = f"\nREUSABLE CHARACTER: {reference}\nSHOT: {prompt}"
                prompt = f"{prompt}. Preserve the exact established character identity from the reusable character line; only the pose, expression and action change. No unrelated people or stock-photo substitutions."
            elif character in CHARACTERS and character != "none":
                character_note = f"\nREUSABLE CHARACTER: {character}"
            scene.image_prompt = prefix + prompt + character_note


def notes_markdown(script: Script, plan: list[dict[str, Any]]) -> str:
    lines = [f"# Director's Notes — {script.title}", "", "Background production plan. These are concise, user-facing decisions, not private chain-of-thought.", ""]
    for beat in plan:
        i = beat["scene"]
        narration = script.scenes[i - 1].narration if 0 < i <= len(script.scenes) else ""
        lines += [
            f"## Scene {i}: {beat.get('purpose', 'beat').title()}",
            f"- **Narration:** {narration}",
            f"- **Visual:** {beat.get('visual_mode', 'illustration').title()} — {beat.get('visual_prompt', '')}",
            f"- **Reusable character:** {beat.get('character', 'none')}",
            f"- **Camera:** {beat.get('camera', 'static').replace('_', ' ').title()}",
            f"- **Transition:** {beat.get('transition', 'cut').replace('_', ' ').title()}",
            f"- **Sound accent:** {beat.get('sound', 'none').title()}",
            f"- **Why:** {beat.get('reason', '')}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
