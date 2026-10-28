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

from .fantasy_video_library import FANTASY_VISUAL_RULES
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
    if style != "fantasy" and (
        any(phrase in narration.lower() for phrase in abstract_phrases)
        or (generic_query and len(narration.split()) <= 8)
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


def _choose_reel_treatment(script: Script) -> tuple[str, str]:
    """Choose one lead visual treatment automatically; content format remains independent."""
    topic = f"{script.topic} {script.title} " + " ".join(
        f"{scene.narration} {scene.search_query} {scene.image_prompt}" for scene in script.scenes
    )
    text = topic.casefold()
    story_words = ("story", "mystery", "secret", "disappear", "door", "stranger", "family", "fear",
                   "character", "dragon", "ghost", "betray", "revealed", "nightmare")
    tutorial_words = ("how to", "tutorial", "tool", "app", "software", "ai ", "artificial intelligence",
                      "workflow", "setup", "step", "website", "money", "business", "creator", "youtube",
                      "instagram", "shorts", "reel", "product", "feature", "system")
    if script.style in ("fantasy", "story", "chat", "reddit") or any(w in text for w in story_words):
        treatment = "character_stage"
        instructions = (
            "LEAD VISUAL TREATMENT: CHARACTER STAGE. Keep one coherent environment and a stable recurring "
            "host/character design when the subject supports it. Use a medium/wide establishing shot, then "
            "subtle push-ins and reaction/detail shots. The character must serve the topic, never be a random "
            "person inserted as filler. Use only the characters and actions justified by the script."
        )
    elif script.style in ("facts", "listicle", "myth", "quiz", "explainer", "news") or any(w in text for w in tutorial_words):
        treatment = "floating_panels"
        instructions = (
            "LEAD VISUAL TREATMENT: FLOATING INFORMATION PANELS. Compose a clean creator-style explainer: "
            "one consistent visual world, a clear subject/background, and room for large separate information "
            "cards to be composited over it. Plan one card per idea, not a wall of tiny text. Use short exact "
            "phrases, one key number or claim per card, generous margins, strong hierarchy, and a deliberate "
            "sequence: hook, evidence/steps, contrast or reveal, payoff, concise CTA when appropriate. Keep "
            "the main subject visible around cards. Do not ask image models to render readable text; the renderer "
            "must add typography separately. Avoid unrelated stock people, clutter, and repetitive wallpaper."
        )
    else:
        treatment = "kinetic_typography"
        instructions = (
            "LEAD VISUAL TREATMENT: KINETIC TYPOGRAPHY. Let a few large, sharply legible phrases carry the "
            "message over relevant footage or a restrained animated gradient. Reveal words in timed beats, "
            "emphasize only the key number/contrast, and vary scale and position with safe margins. Keep each "
            "screen to one message; use imagery as evidence rather than decoration. No tiny UI, text walls, "
            "random photos, or constant motion."
        )
    return treatment, instructions


def direct(script: Script, llm: Any = None) -> list[dict[str, Any]]:
    """Plan each scene; one structured LLM call, with a deterministic local fallback."""
    treatment, treatment_rules = _choose_reel_treatment(script)
    fallback = [_fallback(scene, i, script.style) for i, scene in enumerate(script.scenes)]
    for beat in fallback:
        beat["visual_treatment"] = treatment
        beat["reason"] = (beat.get("reason", "") + f" Lead treatment: {treatment}.")[:240]
    if llm is None or not script.scenes:
        return fallback
    payload = {
        "topic": script.topic,
        "title": script.title,
        "style": script.style,
        "automatically_selected_visual_treatment": treatment,
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
        + treatment_rules + "\n" +
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
    if script.style in ("auto", "facts", "story", "myth", "quiz", "explainer", "news", "listicle"):
        system += (
            "\n\nHYBRID / FACTUAL VISUAL EDIT RULES:\n"            "- Match the visible subject to the exact sentence, not merely the overall topic.\n"
            "- Use real moving footage for real-world events, animals, places, objects, experiments, and processes when a relevant match exists.\n"
            "- Use photos only for moments where a still is genuinely appropriate; use diagrams for a mechanism/comparison that needs explanation, not as generic decoration.\n"
            "- Vary shot scale and visual purpose: establishing context, close-up evidence, action/process, comparison, consequence, payoff. Do not repeat the same broad shot or wallpaper in consecutive scenes.\n"
            "- Search phrases must contain concrete nouns and visible actions or context. Do not use abstract narration phrases as the only stock query.\n"
            "- If the exact footage cannot be found, choose an honest relevant alternative; never imply unrelated footage depicts the claim. A gradient is reserved for brief connective beats only.\n"
        )
    if script.style == "fantasy":
        system += "\n\n" + FANTASY_VISUAL_RULES + "\n\nFor every fantasy scene choose visual_mode=footage and start visual_prompt with FANTASY CINEMATIC FOOTAGE:. Never request drawn/vector art. The user wants the supplied reference-video grammar: premium cinematic fantasy film frames, coherent consecutive shots, stable original characters, actual dialogue coverage, motivated action and reaction shots—not a slideshow, random people, or unrelated stock photos.\n\nREFERENCE-MATCH RULES (highest priority):\n- ACTING, NOT CAMERA ADDRESS: characters look at the person they are speaking to, or at the relevant threat/prop. Never stare into the lens or pose for a portrait. Use eye-lines, over-the-shoulder coverage, side profiles, three-quarter angles and two-shots.\n- SPEAKER VISIBLE THROUGH THE LINE: while each line plays, keep the named speaker visibly on screen (prefer medium close-up or over-the-shoulder with face readable). Perfect lip-sync is not required, but never cover dialogue with unrelated stills or only the other character. Hold the speaker shot through the line, then cut to reaction/action.\n- STILL-IMAGE MOTION: when the source is a still, use subtle camera movement (slow push-in, lateral drift or gentle parallax-like depth where supported). This is camera movement, not character animation; never imply the character itself moves unless source footage actually shows it.\n- Treat the entire script as ONE continuous short film. Build every scene from the immediately preceding beat; preserve location, time of day, light direction, screen direction, costumes, props, injuries, weather and character positions unless the script explicitly changes them. Do not reset the scene visually between lines.\n- Every visual_prompt must specify: shot size/angle, who is in frame, the exact physical verb/action, where each character stands or looks, the important prop, and the emotional beat. Do not use vague prompts like 'epic fantasy scene' or a portrait unrelated to the spoken line.\n- For dialogue: frame the named current speaker clearly with visible face/mouth and the correct emotion; do not depict the listener speaking at the same time. If both characters are needed, use a readable over-the-shoulder/two-shot with only the current speaker's mouth active. Follow the line with a separate listener reaction shot in the next scene when the script supports it.\n- For action, cover cause -> motion -> impact/result -> reaction across consecutive scenes when the script includes enough beats. Keep the same weapon/object, hand, direction, distance and positions; never skip straight from a threat to an unrelated portrait.\n- For the hooded mystery character, use a fitted hooded shirt/hoodie with the hood gathered around the neck/chest, hands visible for gestures, and no long cloak covering the legs unless the story explicitly says so. Preserve that exact design every time he appears. Hide identity with shadow/framing, not by randomly changing his face or clothes.\n- Generate original fictional characters, not real-person photos or stock portraits. Keep each character's exact face, age, hair, skin/scales/fur, build, clothing/armor, palette and signature props consistent. If a character-reference image is available, use it as the generation reference; otherwise repeat the exact textual identity description and never claim an image reference was used.\n- Do not invent extra people, creatures, weapons, locations or visual beats. Do not put readable text, fake subtitles, logos, collage or split screens into generated imagery; captions are added separately.\n- Keep the reference's cinematic realism: grounded photorealistic CGI/live-action fantasy frame, physically plausible lighting and materials, expressive faces, readable action, restrained VFX, vertical 9:16 composition with faces/hands/props above the caption zone.\n- Align cuts to measured TTS duration/word timestamps when available; never start the next speaker before the current line ends. Do not change rendering or runtime workflow."
    try:
        result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), schema=DIRECTOR_SCHEMA,
                                   max_tokens=min(4500, 500 + len(script.scenes) * 150), temperature=0.35)
        planned = result.get("scenes", [])
        by_scene = {int(row.get("scene", 0)): row for row in planned if isinstance(row, dict)}
        merged = []
        for i, base in enumerate(fallback, 1):
            row = by_scene.get(i, {})
            clean = dict(base)
            clean["visual_treatment"] = treatment
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
        for name in script.cast[:6]:
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
                            fantasy_refs[name] = identity[:600]
                            break
                if name in fantasy_refs:
                    break
            if name not in fantasy_refs:
                # Deterministic per-episode designs: if the script model omits a character bible,
                # do not reuse the same red-mage/silver-elf pair in every video.
                looks = (
                    "young adult human spell-thief, one dark braid threaded with copper rings, amber eyes, soot-black leather coat, rust-red scarf, a brass key on a chain",
                    "young adult elven duelist, cropped white hair, dark green eyes, charcoal scale armor, one broken silver ear cuff, narrow black-steel sword",
                    "weathered dwarf engineer, braided ash-brown beard, one cloudy eye, indigo work leathers, bronze shoulder plates, a compact rune-powered hand tool",
                    "young adult desert ranger, warm brown skin, shaved sides and long black topknot, turquoise eyes, sand-colored layered wraps, cobalt sash, curved moonsteel blade",
                    "elderly witch with deep umber skin, silver locs bound in bone beads, pale gold eyes, plum velvet coat, crow-feather mantle, a carved rowan staff",
                    "young half-orc guardian, olive-green skin, one ivory tusk, close-cropped dark hair, moss-green cloak over worn iron armor, a round shield with a cracked sun emblem",
                    "androgynous sky-islander scout, freckled face, storm-grey eyes, wind-tangled auburn hair, cream flight jacket, indigo harness, brass spyglass",
                    "small ancient dragon with deep jade scales, a chipped left horn, molten-gold eyes, soot-dark wing tips, a blue ceremonial collar with one missing gem",
                )
                seed_text = f"{script.topic}|{name}|{len(fantasy_refs)}".casefold()
                seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_text))
                look = looks[seed % len(looks)]
                fantasy_refs[name] = f"original fictional character; {look}; cinematic live-action fantasy realism; exact same face, costume, colors and props in every shot"
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
                raw_speaker = str(getattr(scene, "speaker", "A")).upper()
                speaker_index = ord(raw_speaker) - ord("A") if len(raw_speaker) == 1 and raw_speaker in "ABCDEF" else 0
                visible_name = script.cast[speaker_index] if speaker_index < len(script.cast) else (script.cast[0] if script.cast else "the protagonist")
                # Keep identity and shot instructions on separate lines so the media layer can
                # cache the canonical identity key while generating a different action each time.
                prompt = prompt.removeprefix("FANTASY CINEMATIC FOOTAGE: ").strip()
                visible_names = [visible_name]                # When a shot includes the other named lead, carry that character's fixed
                # identity too; otherwise the generator tends to invent a new face in two-shots.
                scene_context = " ".join((
                    prompt,
                    str(getattr(scene, "narration", "") or ""),
                    str(getattr(scene, "search_query", "") or ""),
                ))
                for candidate in script.cast[:6]:
                    if candidate not in visible_names and re.search(r"\b" + re.escape(candidate) + r"\b", scene_context, re.IGNORECASE):
                        visible_names.append(candidate)
                identity_lines = "\n".join(
                    f"REUSABLE CHARACTER: {name} — {fantasy_refs.get(name, name + ' — original fictional fantasy character; stable design')}"
                    for name in visible_names
                )
                character_note = (
                    f"{identity_lines}\n"
                    f"SHOT: {prompt}. Preserve every listed character's exact established identity; only pose, expression and action change. "
                    "No unrelated people, extra faces, or stock-photo substitutions."
                )
            elif character in CHARACTERS and character != "none":
                character_note = f"\nREUSABLE CHARACTER: {character}"
            scene.image_prompt = (prefix + character_note) if script.style == "fantasy" else (prefix + prompt + character_note)


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