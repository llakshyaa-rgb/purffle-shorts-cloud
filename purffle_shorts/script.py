"""AI scriptwriting: one LLM call returns the whole Short — hook, scenes with per-scene footage
queries, title, description, hashtags and tags — so everything describes the same video.

A second call, the Script Doctor, scores the draft for retention and rewrites what is weak. Scripts can
also be translated for multi-language channels, and two-speaker formats (dialogue, chat) carry a
speaker per scene so each line gets its own voice."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .config import Settings
from .utils import clean_narration, emphasis_key, strip_emoji, truncate_words

log = logging.getLogger("purffle")

LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi", "ur": "Urdu",
    "ar": "Arabic", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Mandarin)",
    "id": "Indonesian", "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "vi": "Vietnamese", "th": "Thai",
    "fil": "Filipino", "uk": "Ukrainian", "sv": "Swedish",
}

STYLES = {
    "auto": "A smart hybrid story blending whichever techniques fit: curiosity hook, narrative tension, surprising facts, myth-check, countdown, analogy, explanation and payoff. Do not force every technique; combine naturally.",
    "fantasy": "An original cinematic fantasy mini-movie told entirely through characters speaking to each other, with no narrator and no voice-over. Build a coherent emotional arc with a small recurring cast, distinct voices, concrete magical stakes, visible actions and reactions, and a satisfying payoff. Open on an ordinary, specific moment before the extraordinary truth emerges; let relationships and choices drive the plot. This is explicitly fictional: do not present invented lore as real-world fact. Do not imitate any named copyrighted franchise or existing character; create original worlds and designs.",
    "facts": "Rapid-fire surprising facts about one specific subject, each one more surprising than the last.",
    "story": "A tight true story: a hook that opens a loop, rising tension, then a satisfying twist or payoff.",
    "listicle": "A top-3 countdown (number 3, number 2, number 1) where number 1 is the most surprising.",
    "myth": "Myth vs. fact: state a belief most people hold, then bust it with clear evidence.",
    "quiz": "Open with a question the viewer will try to answer, build suspense, reveal the answer at the end.",
    "motivational": "A punchy real-life micro-story or principle that ends with a memorable one-line takeaway.",
    "news": "Explain a current story neutrally: what happened, why it matters, what happens next.",
    "explainer": "Explain one idea so a 12-year-old gets it, using one vivid analogy.",
    "dialogue": "A fast back-and-forth between two people, A and B. A is curious or skeptical and asks; B "
                "answers with surprising, specific facts. The last line is a punchline.",
    "chat": "A text-message conversation between A and B that tells a gripping story in real time, with "
            "a twist near the end. Written exactly like real texts: short, casual, emotional.",
    "reddit": "A first-person story told like a viral Reddit post (a confession, an 'am I wrong?' or a 'today I "
              "messed up'). Scene 1 is the post's title, read aloud; the rest is the story in casual first "
              "person, building to a twist or an update at the end.",
}
MULTI_SPEAKER = {"dialogue", "chat", "fantasy"}
CARD_STYLES = {"reddit"}                  # single narrator, but the cast names the community and the poster
AUTO_STYLES = ["facts", "facts", "story", "story", "listicle", "myth", "quiz", "explainer"]
DEFAULT_CAST = {"dialogue": ["Alex", "Sam"], "chat": ["Unknown", "Me"], "reddit": ["r/stories", "u/throwaway"], "fantasy": ["Mira", "Kael"]}

# YouTube video category ids
CATEGORIES = {
    "education": "27", "science": "28", "entertainment": "24", "people": "22", "howto": "26",
    "news": "25", "gaming": "20", "autos": "2", "sports": "17", "travel": "19", "comedy": "23",
    "film": "1", "music": "10", "pets": "15",
}

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]},
        "narration": {"type": "string"},
        "search_query": {"type": "string"},
        "image_prompt": {"type": "string"},
        "emphasis": {"type": "array", "items": {"type": "string"}},
        "emoji": {"type": "string"},
    },
    "required": ["speaker", "narration", "search_query", "image_prompt", "emphasis", "emoji"],
    "additionalProperties": False,
}

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "title": {"type": "string"},
        "hook_text": {"type": "string"},
        "cast": {"type": "array", "items": {"type": "string"}},
        "scenes": {"type": "array", "items": SCENE_SCHEMA},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["topic", "title", "hook_text", "cast", "scenes", "description", "hashtags", "tags", "category"],
    "additionalProperties": False,
}

# The Script Doctor grades six things separately (0-10 each); the retention score is their weighted sum.
# Separate, concrete questions give steadier and more explainable scores than one overall number.
RUBRIC = {
    "hook": (0.30, "the first line stops the scroll within 2 seconds and makes a clear, honest promise"),
    "curiosity": (0.15, "specific open loops make the viewer need the next beat; suspense is earned, not vague"),
    "pacing": (0.15, "no filler, repeated meaning, redundant wording, or slow moment; every line advances the video"),
    "specificity": (0.15, "distinct concrete details, actions, and images instead of vague claims or decorative adjectives"),
    "payoff": (0.15, "the ending fulfils the opening and makes the reveal understandable, causally earned, and consequential"),
    "accuracy": (0.10, "factual claims are well established; fiction is internally coherent and does not pass invention off as fact"),
}
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "integer"} for k in RUBRIC},
    "required": list(RUBRIC),
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "rubric": RUBRIC_SCHEMA,
        "issues": {"type": "array", "items": {"type": "string"}},
        "script": SCRIPT_SCHEMA,
    },
    "required": ["rubric", "issues", "script"],
    "additionalProperties": False,
}
SCORE_SCHEMA = {
    "type": "object",
    "properties": {"rubric": RUBRIC_SCHEMA, "issues": {"type": "array", "items": {"type": "string"}}},
    "required": ["rubric", "issues"],
    "additionalProperties": False,
}
REVIEW_TEMPERATURE = 0.2  # grading should be repeatable; writing stays creative

# Slightly above typical narration pace so generated scripts have enough spoken material;
# the final runtime still follows measured TTS duration rather than silently padding audio.
WORDS_PER_SECOND = 4.0


@dataclass
class Scene:
    narration: str
    search_query: str
    image_prompt: str = ""
    speaker: str = "A"
    emphasis: list[str] = field(default_factory=list)  # words of the narration highlighted on screen
    emoji: str = ""                                    # one emoji popped in above the captions


@dataclass
class Script:
    topic: str
    title: str
    hook_text: str
    scenes: list[Scene]
    description: str
    hashtags: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    category: str = "education"
    style: str = "facts"
    language: str = "en"
    cast: list[str] = field(default_factory=list)
    score: int | None = None           # Script Doctor retention score (0-100) of this exact script
    review: list[str] = field(default_factory=list)  # what the Script Doctor found
    rubric: dict[str, int] = field(default_factory=dict)  # the sub-scores behind ``score`` (0-10 each)

    @property
    def narration(self) -> str:
        return " ".join(s.narration for s in self.scenes)

    @property
    def category_id(self) -> str:
        return CATEGORIES.get(self.category, "27")

    @property
    def multi_speaker(self) -> bool:
        return any(s.speaker != "A" for s in self.scenes)

    def speaker_name(self, speaker: str) -> str:
        i = ord(speaker.upper()) - ord("A") if len(speaker) == 1 and speaker.upper() in "ABCDEF" else 0
        return self.cast[i] if i < len(self.cast) else speaker

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Script:
        known = {f.name for f in fields(cls)}
        scene_keys = {f.name for f in fields(Scene)}
        d = {k: v for k, v in dict(d).items() if k in known}
        d["scenes"] = [Scene(**{k: v for k, v in s.items() if k in scene_keys}) for s in d.get("scenes", [])]
        return cls(**d)


def pick_style(settings: Settings, source: str) -> str:
    if settings.style in STYLES:
        return settings.style
    if source == "trending":
        return "news"
    if source in ("wikipedia", "reddit", "rss"):
        return "story"
    return "auto"


def scene_plan(style: str, target_seconds: int) -> tuple[int, int]:
    """Return a narration word target and a practical shot/scene target for the runtime."""
    words = int(target_seconds * WORDS_PER_SECOND)
    if style == "chat":
        suggested = round(target_seconds / 2.8)
    else:
        # Non-chat scenes are usually a few seconds long; keep enough distinct visuals
        # without asking a normal narration to fragment every clause into a cut.
        suggested = round(target_seconds / 3.5)
    return words, max(3, min(16, suggested))


def scene_count_bounds(style: str, target_seconds: int) -> tuple[int, int]:
    """Reasonable scene-count bounds: enough distinct shots to keep a Short visually moving."""
    _, target = scene_plan(style, target_seconds)
    minimum = max(3, target - 2)
    maximum = min(16 if style != "chat" else 20, target + 3)
    return minimum, maximum


def _format_rules(style: str, target_seconds: int = 40) -> str:
    if style == "dialogue":
        return (
            "- Each scene is ONE line spoken by ONE person. Alternate speakers: A, B, A, B...\n"
            "- cast: two short first names, [name of A, name of B]. Nobody says their own name.\n"
        )
    if style == "fantasy":
        return (
            "- Invent a fresh cast for this exact prompt and episode. Add characters only when they improve plot, stakes, mystery, conflict, or payoff; never reuse a default cast or fill slots with generic roles. Use as many meaningful speaking characters as the story needs, normally 3–6 when appropriate, with speaker IDs A–F. Give each character a distinct name, appearance, goal, relationship, and reason to be present.\n"
            "- Create a CHARACTER BIBLE before drafting. Define each main character's species, approximate age, face, eyes, hair/fur/scales, build, clothing/armor, signature prop, and color palette. Keep these details unchanged throughout the Short.\n"
            "- Every relevant image_prompt must begin with: REUSABLE CHARACTER: NAME — [fixed appearance, outfit, palette, signature prop] then a new line: SHOT: [the exact action and camera framing for this scene]. Repeat the identical identity description whenever that character reappears; change the shot to match the story beat. Never use random real-person photos or unrelated stock portraits as canonical characters.\n"
            "- ZERO NARRATION OR VOICE-OVER. Every spoken line is direct dialogue spoken by a visible character to another character, or a short natural self-directed line if alone. Never narrate events or describe what the viewer can already see. Show time passing visually and reveal backstory through believable dialogue. If a beat cannot be expressed naturally as dialogue, show it silently.\n"
            "- Each scene contains one concise, natural line spoken by the character in its speaker field. A is cast[0], B is cast[1], C is cast[2], D is cast[3], E is cast[4], and F is cast[5]. Every named speaker must have a cast entry, and dialogue must respond to what others just said or did.\n"
            "- Follow the user's actual topic and intent; do not force a preset plot onto every fantasy prompt. For an open-ended hidden-dragon-child/family-origin prompt, a possible arc is ordinary home and unusual child, visual growth over time, an external danger, the parents revealing they are dragons through direct dialogue, the child making a consequential choice, and an emotional payoff. For other topics, invent an equally specific arc faithful to the requested premise.\n"
            "- Storyboard a filmable chain of cause and effect: establish location and characters, trigger, reaction, decision, physical action, consequence, and payoff. If a dagger is thrown, show the throw, its path or impact when important, and the target's reaction in consecutive shots. Track props, positions, and movement. No random portraits or disconnected mood images.\n"
            "- SEQUENTIAL CINEMATIC FANTASY (priority): this is a tiny film, not a slideshow. Choose one consistent visual medium and style for the entire video (cinematic 3D animated fantasy unless the prompt clearly calls for another style). Every shot must continue the exact moment, location, cast, and action established by the previous shot. Never introduce a new face, outfit, creature, weapon, room, or visual style without a story reason. Keep the mysterious character visually consistent while withholding only the identity details the plot intends to conceal.\n"
            "- SPEAKING-SHOT RULE: each dialogue scene holds on the character who is actually saying that line, framed so the face/mouth and emotion are readable; do not montage over dialogue. After the line finishes, cut to the listener's reaction or the next motivated action. Keep dialogue short enough to perform naturally, but preserve the complete causal story.\n"
            "- ACTION COVERAGE RULE: for any important movement, storyboard the physical sequence in order: who initiates it, the moving object/body and direction, what it hits or changes, then the affected character's reaction. For a thrown dagger, use consecutive matching shots for throw -> flight/impact -> target reaction when the moment matters; keep the same dagger, hand, screen direction, and positions. Never replace an action with unrelated character portraits.\n"
            "- VISUAL CONTINUITY LOCK: repeat each recurring character's same concise identity description verbatim; repeat essential costume/prop/location cues in every relevant prompt. Each prompt describes ONLY its own shot, with shot size/angle, visible action, and character placement. No collage, split-screen, text, random extras, or unrelated stock-photo substitutions.\n"
            "- During dialogue, clearly frame the current speaker and hold that shot until the line finishes; do not show the listener apparently speaking. Cut to the listener's reaction only after the line. Give important physical actions their own shots.\n"
            "- ACTING, NOT CAMERA ADDRESS: characters look at the person they are speaking to, or at the relevant threat/prop. Never stare into the lens or pose for a portrait. Use eye-lines, over-the-shoulder coverage, side profiles, three-quarter angles and two-shots.\n"
            "- SPEAKER VISIBLE THROUGH THE LINE: while each line plays, keep the named speaker visibly on screen (prefer medium close-up or over-the-shoulder with face readable). Perfect lip-sync is not required, but never cover dialogue with unrelated stills or only the other character. Hold the speaker shot through the line, then cut to reaction/action.\n"
            "- STILL-IMAGE MOTION: when the source is a still, use subtle camera movement (slow push-in, lateral drift or gentle parallax-like depth where supported). This is camera movement, not character animation; never imply the character itself moves unless source footage actually shows it.\n"
            "- Keep geography, screen direction, costumes, props, lighting logic, and character designs consistent across adjacent shots. Vary shot types only when the story beat calls for them.\n"
            "- Synchronize cuts to measured voice duration or word timestamps when available. Never begin the next speaker before the current line ends. Captions must match the spoken line and timing.\n"
            "- cast must be an ordered list of original speaking character names with stable A–F mapping. Every scene speaker must identify exactly who says that line; do not cap the story at two characters when the premise needs more.\n"
        )
    if style == "chat":
        return (
            "- Each scene is ONE text message (max 18 words) from A or B. Messages may be 1–3 words. Consecutive messages from the same person are fine.\n"
            "- cast: [contact name shown at the top of the chat (e.g. Mom, Unknown Number, Jake), 'Me']. B is \"Me\", the phone's owner.\n"
            "- No emojis. The story must be fiction that is clearly plausible, never about real people.\n"
        )
    if style == "reddit":
        return (
            "- speaker is always 'A': the poster, telling it in the first person.\n"
            "- Scene 1 narration is ONLY the post title: a first-person question or confession, max 16 words. Spell things out for the voice (for example, 'Am I wrong for', not 'AITA'); no abbreviations like TIFU.\n"
            "- cast: [a fitting community name starting with r/ (e.g. r/confessions, r/pettyrevenge), a made-up username starting with u/]. hook_text is not shown for this format.\n"
            "- Fiction written like a real post: invent every person, never name real people. No emojis.\n"
        )
    return '- speaker is always "A" (one narrator). cast: ["Narrator"].\n'

def build_prompt(subject: str, style: str, settings: Settings, avoid: list[str], context: str = "",
                 insights: str = "", keywords: str = "", creative_brief: str = "") -> tuple[str, str]:
    lang = LANGUAGES.get(settings.language, settings.language)
    words, n_scenes = scene_plan(style, settings.target_seconds)
    system = (
        "You are an exceptional human-sounding short-form scriptwriter, storyteller, and line editor. "
        "Your scripts feel clear, specific, emotionally engaging, and effortless to understand when heard once. "
        "You build a coherent spoken script before dividing it into scenes. Every sentence has a purpose; "
        "no robotic phrasing, empty hype, unexplained jargon, filler, or disconnected fact-dumps. "
        "You reply with one valid JSON object and nothing else."
    )
    avoid_block = ""
    if avoid:
        avoid_block = "Do NOT repeat any of these recent videos:\n" + "\n".join(f"- {a}" for a in avoid[:40]) + "\n\n"
    ctx = f"Background information (use it, don't invent beyond it):\n{context.strip()}\n\n" if context else ""
    ins = f"{insights.strip()}\n\n" if insights else ""
    ins += f"{keywords.strip()}\n\n" if keywords else ""
    brief = (f"CREATOR CREATIVE DIRECTION (follow unless it conflicts with factual accuracy or safety):\n{creative_brief.strip()[:1000]}\n\n" if creative_brief.strip() else "")
    hybrid_rules = ""
    if style == "auto":
        hybrid_rules = """HYBRID STORY + FACTS QUALITY BAR (mandatory):
- Choose ONE clear central question or promise. Build one connected progression from hook to evidence/events to payoff; do not switch to unrelated facts just to vary the format.
- Use factual details only when they strengthen the same thread. If the topic is a true story, distinguish confirmed facts from uncertainty. Never invent statistics, quotes, dates, study results, or citations.
- Every scene must add a new beat: specific evidence, a concrete example, a consequence, a meaningful comparison, or the payoff. Remove repeated claims and generic bridge lines.
- Each scene's search_query must name the visible subject plus its action/place/context. Each image_prompt must describe what the viewer should actually see in that beat, not merely repeat the topic or say "cinematic footage".
- Prefer relevant moving footage for real-world subjects when a credible match exists. Use close-ups, wide shots, processes, comparisons, maps, or simple diagrams only when they clarify the spoken beat. Avoid repeated topic wallpaper, random people, unrelated stock clips, and decorative gradients. A still is a fallback when it is the best available match; subtle camera movement is not character animation.
- Keep visuals aligned with the spoken line at every cut. A scene about a specific object, action, place, or experiment must show that subject/action; a general atmosphere shot is not enough. If the exact visual is unavailable, use the closest honest explanatory visual rather than implying unrelated footage is evidence.
- End by resolving the opening question or delivering the promised reveal. No generic engagement-bait ending, fake cliffhanger, or extra fact that does not complete the thread.

"""
    user = f"""Write one YouTube Short.

Subject: {subject}
If the subject is broad, choose ONE specific, surprising, lesser-known topic inside it.
Format: {STYLES[style]}
Audience: {settings.audience}
Language: {lang} for narration, title, hook_text, cast and description. search_query and image_prompt are ALWAYS English.

{ctx}{ins}{brief}{hybrid_rules}{avoid_block}Rules:
- HARD LENGTH TARGET: write about {words} spoken words for a {settings.target_seconds}-second video. Treat this as a requirement, not a loose suggestion; do not finish early after making the point. The measured voice duration should land close to the requested duration. Aim for {n_scenes} scenes; acceptable range is {scene_count_bounds(style, settings.target_seconds)[0]}–{scene_count_bounds(style, settings.target_seconds)[1]}. Scene count is a real visual-editing requirement, not a suggestion to ignore. For a 30–60 second Short, do NOT compress the entire script into 3–5 oversized scene boxes. Use roughly one scene per 2–3 seconds where the narration supports it: usually one sentence or one compact spoken beat per scene. Keep the full narration length; divide it across more scenes instead of cutting text. Each scene needs its own specific visible action, evidence, reaction, transition, or payoff. Do not create empty/filler scenes, split every clause mechanically, or combine distinct visual beats into one box.
{_format_rules(style, settings.target_seconds)}- For fantasy scenes, use speaker A–F as real character attribution, not a single narrator placeholder; map each letter to the matching cast entry. Every fantasy image_prompt must start with a reusable character identity line and a separate exact-shot line. Keep every named character recognizable whenever they reappear, with story-specific identity details; show physical action, meaningful reactions, and consequences in consecutive shots. Pixabay may provide reference candidates but is not an AI image generator. Prefer the configured AI image provider for original reference portraits; never substitute unrelated stock people for original characters. Keep the central conflict resolved before an optional sequel tease.
- Scene 1 is the hook: make a specific, vivid opening that stops the scroll in under 2 seconds. NEVER default to "Did you know?" or repeat a generic hook formula. Vary openings naturally: an unusual concrete detail, a counterintuitive claim, a vivid moment, a sharp question, a tiny mystery, or a direct scene-setting line. Use "Did you know?" only when it is genuinely the best fit, and never in consecutive/recent videos if avoid titles or insights suggest it was used.
- FACTUAL OPENING (mandatory for facts, myth, quiz, explainer, news, and listicle): do not begin with a disconnected fact dump. In the first one or two spoken lines, establish the claim, common belief, surprising contradiction, or real-world question that makes this topic worth checking. Give the viewer a reason to care or a clear question to resolve, then deliver the evidence. Example structure, not wording to copy: "People say X — but does the evidence actually support it?" / "This sounds impossible, so let's check what really happens." Make the claim specific, avoid vague "you won't believe this" bait, and never imply a fact is false unless the evidence supports that. For rapid-fire facts, use one brief framing hook that connects the facts under a clear question or surprising theme.
- NATURAL NARRATION FLOW (mandatory): for fantasy, use believable dialogue exchanges with reactions and subtext—not narration. For all other styles, write for one continuous spoken performance. Avoid filler and artificial suspense pauses.
- CONTINUITY PASS (mandatory): for fantasy, ensure every line is a direct response or motivated statement in the scene; track what each character knows, their emotional state, props, location, and time across shots. Do not let characters know facts they have not learned. For other styles, keep narration coherent and resolve the opening question.
- WRITE THE SCRIPT AS A WHOLE BEFORE SPLITTING IT INTO SCENES. The narration must read like one polished spoken piece from beginning to end, not a set of independent lines. Each scene should continue the thought, answer something raised earlier, or create a clear reason to hear the next scene.
- CREATIVE DESIGN BEFORE PROSE: first choose the single strongest angle for this topic, the viewer's central question, the most interesting evidence or event, and the ending that makes the opening worthwhile. Build one coherent progression suited to the format: opening -> development -> meaningful complication/evidence -> payoff. A list is allowed only when the list itself is the promise; otherwise connect the information into one thread. Avoid "fact babbling", generic AI phrasing, and announcing what the video will explain.
- STORYTELLING DETAIL (without copying any creator or reference video): write a vivid, tightly edited story the viewer can picture as animation. Use concrete actions, distinct objects, visible reactions, and consequences. For every major beat, establish what happens, what the character notices, what they choose, and what changes because of that choice. Make choices cause the next beat. Replace vague summaries with the specific action or detail that makes the moment real. Build tension through evidence and decisions, not repeated words like "creepy", "terrifying", "mysterious", or "suddenly".
- STORY SCENE ARCHITECTURE (mandatory for story/reddit/chat fiction): before writing, privately design a beat map with (1) a concrete hook/question, (2) grounded setup, (3) escalating evidence, (4) a reveal that explains the opening clue, (5) a protagonist decision under a real trade-off, and (6) a consequence that proves the decision mattered. Make each beat change what the character knows, wants, risks, or does. A surprising reveal alone is not a story. The protagonist must actively choose something; that choice must cause the climax and the ending. Give the ending a specific physical or emotional result, not a vague reaction or an unexplained final line. Plant at least two fair clues when the premise is extraordinary, then pay them off. Keep mechanics understandable on one listen. Do not force horror or a twist if a tender, funny, tragic, or realistic resolution is stronger. The viewer should be able to retell what happened, why it happened, what choice was made, and what changed afterward.
- Every video is a COMPLETE STANDALONE PIECE, not an episode in a series. Give it a clear beginning, development, and ending/payoff within the chosen duration. Do not use "part 2", "follow for the next part", or leave the main question unresolved just to bait another view. Do not invent continuity from previous videos unless the topic explicitly asks for it.
- Avoid repetitive engagement bait. Never automatically end with "What do you think?", "Would you have done it?", or "Follow for part 2". End on the story's consequence, reveal, emotional beat, or useful takeaway. Add a brief CTA only when it feels natural and does not weaken the ending; vary it and omit it when the payoff is stronger alone.
- For fictional or dramatized stories, build a visible mini-arc: establish the character and immediate situation, introduce a specific disturbance, show a reaction and a deliberate choice, make that choice change what happens, then deliver a concrete reveal or consequence. The ending must resolve or transform the central mystery, not merely announce something eerie and stop. Make the twist understandable on first listen and plant enough detail that it feels earned. Include a human reaction after the reveal when time allows. For factual stories, keep real-world claims accurate and distinguish documented facts from invented dialogue or reconstruction.
- HARD ENDING CHECK FOR FICTION: the final narration must answer at least one central question raised by the hook and show the immediate consequence of the reveal. Do not end on a new character’s unexplained line, a wave, a door opening, or a fresh mystery alone. If the reveal introduces a person or impossible event, explain one concrete connection to the opening (for example, why they were waiting, how they knew the narrator, or what the narrator recognizes) within the remaining scenes. A sequel may be possible, but the current story must still feel complete.
- FULL EDITORIAL PASS (mandatory): edit the whole piece as an editor, not a proofreader. Cut any line that adds no new information, image, feeling, causal link, or useful pause. Replace abstract claims with specific evidence. Remove duplicate beats, redundant explanations, empty setup, unearned twists, over-explaining, and lines that merely describe what the viewer can already see. Repair grammar, reference clarity, chronology, and transitions. Do not preserve a weak line just because it sounds cinematic.
- DISTINCTIVE HUMAN NARRATOR VOICE (mandatory for every format): sound like a clever, curious person telling one viewer something fascinating—not a documentary robot, textbook, or trailer. Use conversational phrasing and contractions where natural. Be confident without pretending certainty. Use small, dry, relatable jokes occasionally when they fit the subject, but never force a joke into every scene or undercut a serious, tragic, frightening, or sensitive moment. The joke should make the idea relatable, not distract from it.
- QUESTION / HYPOTHETICAL OPENINGS: when the subject is a question or "what if" scenario, say the actual question naturally in the opening (for example, "What would happen if Earth stopped moving for one second?"). Then meet the viewer where they are: briefly voice the intuitive or funny assumption someone might have ("You might think we'd barely notice"), before clearly correcting it with the real mechanism and consequences. This expectation -> correction -> explanation pattern should feel conversational, not like a rigid template. Don't use a fake joke or a straw-man belief just to set up the reveal.
- FACTS AND EXPLAINERS: make the viewer understand both WHAT happens and WHY. Use a clean progression: question or surprising claim -> relatable first impression (sometimes a quick joke) -> accurate explanation -> concrete consequence/example -> satisfying takeaway. Add a simple analogy only when it genuinely clarifies. Keep uncertainty and scientific limits accurate; never exaggerate danger or state speculation as fact for drama. Explain the causal chain in plain language (what moves, why, and what people would experience), not merely a string of dramatic claims.
- STORIES: tell the events as a compelling person would recount them aloud, with natural transitions, clear chronology, specific actions and human reactions. Let the narration feel close to the character's experience; use first person only when the story is meant to be told by that character, otherwise use a clear storyteller's voice. Let dialogue or a small moment of humor appear only when believable for that character and situation. Don't interrupt an emotional or scary scene with a random joke. Build suspense from what the character sees, thinks, and does, and make the final reveal/consequence understandable on first listen.
- VARIETY, NOT A FORMULA: not every video needs a question, joke, or "you might think" line. Adapt the voice to the subject, mood, and format. Use the pattern when it improves clarity, then move on. Avoid repeating the same opening, joke shape, transition, or payoff across consecutive videos.
- Avoid canned AI phrases such as "little did they know", "but here's the twist", "what happened next changed everything", "you won't believe", and "and that's when things got interesting" unless the specific story genuinely earns that wording. Do not use dramatic language as a substitute for information.
- Do not force every sentence to be a hook. Use rhythm: a few short punchy lines, then a clear explanatory line when needed, then move forward. The script should sound natural at normal speaking speed, not like every line is a trailer voice-over.
- SPOKEN-LINE TEST (mandatory): for fantasy, read every line as an actor speaking to another character in that exact moment. If it sounds like a narrator, plot summary, exposition dump, or a line written only for the audience, rewrite it as believable dialogue or remove the spoken line and communicate the beat visually. For other styles, ensure narration sounds natural aloud.
- NATURAL VOICE, NOT CHATBOT VOICE: avoid generic narrator scaffolding ("Here is the thing", "little did I know", "what I didn't realize", "but everything was about to change", "the truth was shocking"), vague emotional summaries ("a strange mix of gratitude and loss"), and tidy moral statements ("he learned it was about potential") unless the actual story earns them. Show emotion through a specific reaction, choice, remembered detail, or consequence. Don't explain what a scene already makes obvious; don't hide information behind vague poetic wording either.
- STORY DETAILS MUST PAY OFF: a striking object or detail introduced early should matter later. Every reveal must be supported by planted clues and followed by a clear human reaction or consequence. Do not add a second twist just to make the ending sound bigger. The viewer should be able to retell what happened and why after one listen. If an animal, person, or object is revealed inside a sealed or tiny space, make the access and placement physically understandable, and connect its presence to the opening disturbance.
- NO WORD-ECHO OR MEANING-ECHO: never write constructions like "a bustling city ... bustles", "he whispered quietly", "a tiny miniature", or repeat an idea with a synonym in the same beat. Adjectives must add information, not decorate a noun with a duplicate meaning. After drafting, scan every sentence for repeated roots, tautologies, and two clauses that say the same thing; rewrite the weaker wording.
- FICTION LOGIC AUDIT: every extraordinary premise needs a clear internal rule and a causal chain: anomaly -> evidence -> discovery -> choice -> consequence. Track physical geography and scale, elapsed time, object placement, access, and what every character or animal could plausibly know and do. Run a hard plausibility check: could this person or creature survive the stated duration with the available food, water, air, shelter, and injuries? Does its age, appearance, condition, or ability match the time that passed? If not, either plant a clear extraordinary rule and evidence for it, change the duration/condition, or remove the claim. Never patch a contradiction with a hand-wavy line such as 'it survived on crumbs' unless the story establishes a credible supply and explains the rest of its needs. For an impossible or magical story, establish the rule early and apply it consistently; do not pretend magic is ordinary biology. Check that every important object can physically get where it is found, that characters can reach each other, and that the reveal truly explains the opening disturbance. If a person is a future/past version of the protagonist, establish how they know, why they can communicate, what exact danger they are warning about, what happens if the protagonist ignores them, and how the protagonist's choice changes that outcome. Keep the rule consistent; do not invent a new rule only to rescue the ending. Use planted evidence that a viewer can notice before the reveal. If the premise cannot be explained clearly within the runtime, simplify it rather than leave contradictions. Never let "it was his future self" or "they warned him" substitute for a causal explanation. One coherent, earned reveal beats three random twists.
- ORIGINALITY OVER STOCK TWISTS: avoid defaulting to an identical twin, evil clone, future self, shadow under the bed, or "it was inside the room all along" unless the premise specifically earns it with clues. Do not merely replace nouns in a familiar template. Find the most distinctive consequence of this exact premise and build the story around it.
- CONTINUITY BETWEEN SCENES: after drafting, temporarily join all scene narration into one continuous passage and check it without scene numbers or visuals. Repair abrupt time jumps, unexplained motivations, point-of-view changes, repeated information, and lines that only make sense when read as separate cards. Then split it at natural breath and visual changes. Scene boundaries must never dictate sentence structure.
- TITLE / HOOK / STORY ALIGNMENT: the title, on-screen hook, and spoken opening must honestly describe the same central subject. Never pair an unrelated clickbait title with a different story, and never let a visual motif (such as a tiny chef) replace the promised answer or central question.
- For factual, science, explainer, quiz, and hypothetical "what if" videos, make the opening sound like a person talking to a curious friend. If the subject is a what-if question, open with a direct version of the question, such as "So, what would happen if you stayed awake for 24 hours?" Do not force suspenseful metaphors ahead of the actual question.
- Use everyday words first. Avoid dropping unfamiliar scientific or medical names just to sound smart. If a technical term is genuinely useful, explain it in the same breath: say what it is in plain language and what it does, then continue. Example pattern: "Your brain builds up adenosine, a chemical that makes you feel sleepy." Never leave a new term unexplained or stack several unfamiliar terms in one sentence.
- Explain the mechanism, not just the label or outcome: briefly tell viewers what is happening, why it happens, and what they would notice. Prefer a clear real-world comparison when it helps. Keep explanations accurate and short enough to follow while listening.
- Write for someone who has not studied the subject: simple, specific spoken English, contractions where natural, no textbook voice, no jargon parade, no vague phrases like "your brain enters a new state." Do not dumb down the science or invent certainty.
- In facts and quiz formats, the opening question should be answered progressively. Each scene should make the next detail easier to understand, not feel like a separate fact.
- Universal narration quality rules apply to EVERY format and topic, not just science, facts, or hypothetical questions:
  * Sound like a sharp, natural human creator talking to one viewer. Use plain, familiar words, contractions where natural, and varied sentence lengths. Avoid robotic, generic, stiff, over-polished, or article-like prose.
  * Make the first spoken line specific to the topic and immediately worth hearing. Use a direct question when the viewer's question is the point; otherwise use a concrete moment, surprising detail, conflict, or claim. Never force one hook formula onto every video.
  * Explain anything the viewer may not know. If an unfamiliar term, acronym, person, place, event, tool, or concept matters, give a quick plain-language explanation of what it is and why it matters before moving on. Do not stack unexplained names or jargon.
  * Make the logic easy to follow: connect each line to the previous one, explain cause and effect where relevant, and use concrete examples or comparisons when they clarify the point. Do not merely announce labels, list claims, or tease information without delivering it.
  * Match the voice to the format and subject: stories should feel lived-in and build naturally; facts should explain why each fact is surprising; quizzes should ask clearly and reveal a satisfying answer; news should give context without hype; tutorials should state what to do and why; dialogue should sound like real people; motivational scripts should avoid empty slogans.
  * Be specific without becoming dense. Keep one main idea per sentence or breath. Use short transitions only when they help. Cut filler, repeated conclusions, fake suspense, generic phrases, and unnecessary dramatic adjectives.
  * Preserve nuance and accuracy. Do not invent evidence or pretend uncertain claims are proven. If a subject is complicated, explain the key idea simply rather than oversimplifying it into something false.
- Every line earns its place: use concrete details, names, comparisons, actions, and well-established facts. No invented statistics, fake certainty, or sensational claims unsupported by evidence.
- FINAL QUALITY GATE — DO NOT SKIP: run three silent passes before returning JSON. PASS 1, STORY LOGIC: write a one-sentence causal summary; if any step needs "somehow", repair it. Verify every reveal was foreshadowed and every character's decision has a visible consequence. PASS 2, SPOKEN EDIT: read the narration as one continuous spoken piece; remove awkward phrasing, repeated information, overlong sentences, pronoun ambiguity, and lines that only sound dramatic on paper. PASS 3, VIDEO EDIT: check every scene has one distinct visual beat, a usable English footage search, and a specific image prompt; maintain the requested scene density without splitting words mechanically. Then score the draft honestly against hook, progression, clarity, originality, emotional payoff, accuracy, and visual usefulness. Rewrite weak parts before returning JSON. If the premise is too broad, narrow it; if it is too thin, add causal detail, not filler. Never trade coherence for a shock twist.
- CAUSE-AND-EFFECT TEST: every important action must have a reason and consequence. Do not introduce a key, person, object, note, or relationship only in the final line unless the opening planted it or the ending clearly explains it. If a detail is called "missing", establish that it went missing earlier. Pronouns such as he/she/they/her must have an unambiguous referent.
- SCENE COUNT AND EDITING RHYTHM: aim for {scene_plan(style, settings.target_seconds)[1]} scenes (acceptable range {scene_count_bounds(style, settings.target_seconds)[0]}–{scene_count_bounds(style, settings.target_seconds)[1]}). For a 30–60 second Short, 3–5 scene boxes is usually too few: preserve the full narration and split it into distinct, visually useful beats averaging about 2–4 seconds. Do not pad with meaningless cuts; do not merge different actions, evidence, reactions, or reveals into one oversized scene. The final scenes must deliver the promised answer and its consequence.
- VISUAL STORYBOARD QUALITY: write each scene as a distinct shot that changes the information or emotion. The image_prompt should specify subject, visible action, setting, framing, and a useful detail; search_query should describe the actual visible subject, not abstract mood. Avoid reusing the same generic shot across scenes. For fictional animation, keep character identity, clothing, scale, props, and location consistent across the script by repeating concise visual continuity cues in each relevant image_prompt.
- VISUALS ARE YOUR JOB, NOT THE USER'S: choose the visual concept and shot progression yourself from the story. Each image_prompt must be ready to use without the user rewriting it: specify the exact subject and action, environment, shot size/angle, lighting, and relevant continuity details. Give scenes varied compositions (establishing shot, detail insert, reaction, movement, reveal) only when story-appropriate; do not make every scene a generic close-up. For fictional scenes, preserve a stable character/prop design and physical layout. Never invent a separate visual story that contradicts the narration; the visuals should clarify or heighten the exact beat being spoken.
- SCENE-TO-SCENE CONTINUITY: keep point of view, tense, character names, object ownership, and physical geography consistent. Each scene must follow naturally from the previous one. Do not change "I" to "he" without a clear reason, and do not make the narrator know something they could not know.
- ACTION-BY-ACTION STORYBOARD (mandatory for fiction and fantasy): storyboard physical action as consecutive, filmable beats, not a slideshow of character portraits. If a character starts an action (raises a weapon, throws a dagger, opens a door, runs, or grabs an object), show the action beginning, the object/person moving through space when important, the target or impact, and the reaction/consequence in the next appropriate scene. Track who holds each prop before and after every action. Keep the same named characters, clothing, face, weapon, room layout, screen direction, and lighting across connected shots; repeat compact identity and prop cues in every relevant image_prompt. Use a clear shot sequence such as speaker medium shot -> hand/prop insert -> action in motion -> target reaction, only as the narration requires. Never substitute unrelated people, generic portraits, or mood-only images for the narrated event. Hold a speaking character on screen for the full spoken line; place the listener reaction or next action only after that line finishes. Each scene's image_prompt must describe exactly the beat being narrated, not the whole story or a different moment.
- LENGTH AND DENSITY: target the requested runtime without padding. Each scene should contain one clean spoken beat; keep sentences easy to follow on first listen. If the premise needs more explanation than the runtime allows, simplify the premise rather than cutting essential cause-and-effect.
- SELF-REVIEW BEFORE JSON: ask silently: What does the viewer wonder after the hook? What evidence builds the question? What choice changes the situation? What does the reveal prove? What happens because of it? If any answer is missing, revise the script before returning JSON.
- Plan visuals while writing, not afterward. Each scene must depict a distinct, specific thing the viewer can actually see that directly matches its narration. Give every scene a useful, searchable visual subject; avoid repeating the same subject across adjacent scenes unless a deliberate return shot helps the story.
- search_query: 2-5 specific English words describing the exact visible subject of the scene. Prefer concrete objects, organisms, scientific structures, actions, or locations. For science topics, use specific terms such as "Lake Natron flamingos", "sodium carbonate lake shore", or "flamingo feet alkaline water". Avoid vague phrases such as "hidden roles", "glowing brain", "amazing science", or "brain facts". Match the actual narration; never invent visual subjects.
- image_prompt: one vivid English sentence describing a distinct shot for that scene, including subject, action, setting, and camera framing when helpful. No text, labels, or letters in the image.
- Do not make a scene visually generic just because the narration is abstract: use a relevant close-up, real-world example, map, diagram, archival image, or process visualization that communicates the exact point.
- The last scene lands a clear payoff or final consequence. A call to action is optional, never automatic; omit it if it would dilute the ending. Never default to a question or "follow for part 2". For fiction, the final two scenes must explain the reveal and show its immediate consequence; a mysterious line or image by itself is not a complete ending.
- Narration is spoken by a voice: no emojis, hashtags, labels, brackets or stage directions.
- image_prompt: one vivid English sentence depicting the same specific subject and action as search_query. No text, labels, or letters in the image.
- emphasis: the 1-2 words of that scene's narration that carry it (a number, a name, the surprising word), copied exactly as written there. They are highlighted on screen; use [] when nothing stands out.
- emoji: ONE emoji that fits that scene, popped up on screen above the captions (e.g. 🐙 for an octopus, 🤯 for a twist), or "" when none fits. Never put emoji in the narration.
- title: max 60 characters, curiosity-driven but honest, no hashtags, no quotes.
- hook_text: max 6 words shown on screen during the first seconds.
- description: 2-3 sentences plus one question that invites comments. No hashtags.
- hashtags: at least 5 distinct, highly relevant hashtags plus #shorts. Avoid generic spam. tags: 8-15 search keywords.
- category: one of {", ".join(CATEGORIES)}.

Return JSON with keys: topic, title, hook_text, cast, scenes (array of {{speaker, narration, search_query, image_prompt, emphasis, emoji}}), description, hashtags, tags, category."""
    return system, user


def _clean_tag(tag: str) -> str:
    return re.sub(r"[<>\"#]", "", strip_emoji(str(tag))).strip()


def _clean_hashtag(tag: str) -> str:
    t = re.sub(r"[^\w]", "", strip_emoji(str(tag)), flags=re.UNICODE)
    return f"#{t}" if t else ""


def _speaker(value) -> str:
    raw = str(value or "A").strip().upper()
    return raw[0] if raw and raw[0] in "ABCDEF" else "A"


def _handle(name: str, prefix: str) -> str:
    """'confessions' -> 'r/confessions'; spaces become underscores, as in real handles."""
    core = re.sub(r"^/?[ru]/", "", name.strip(), flags=re.I)
    core = re.sub(r"\s+", "_", core).strip("_/") or "anonymous"
    return prefix + core[:22]


_KEY_WORD = re.compile(r"\w", re.UNICODE)
_STOP = {"the", "and", "but", "for", "with", "that", "this", "was", "are", "you", "your", "its", "his", "her",
         "they", "them", "their", "from", "have", "has", "had", "not", "all", "can", "will", "just", "into"}


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF]")


def one_emoji(value) -> str:
    """The first emoji in ``value`` (with its variation selector, skin tone or zero-width-joined parts),
    or "" when there is none."""
    text = str(value or "").strip()
    m = _EMOJI.search(text)
    if not m:
        return ""
    out, i = m.group(), m.end()
    while i < len(text) and (text[i] in "\ufe0f\u200d" or "\U0001F3FB" <= text[i] <= "\U0001F3FF"
                             or (text[i - 1] == "\u200d" and _EMOJI.match(text[i]))):
        out += text[i]
        i += 1
    return out.rstrip("\u200d")


def _emphasis(value, narration: str, limit: int = 3) -> list[str]:
    """The LLM's emphasis words that really occur in the narration (so they can be found when the line is
    spoken), split into single words, without common little words."""
    spoken = {emphasis_key(w) for w in narration.split()}
    out: list[str] = []
    items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    for item in items:
        for w in str(item).split():
            k = emphasis_key(w)
            if k and _KEY_WORD.search(k) and k in spoken and k not in _STOP and k not in out:
                out.append(k)
    return out[:limit]


def normalize(data: dict, subject: str, style: str, language: str) -> Script:
    multi = style in MULTI_SPEAKER
    raw_scenes = data.get("scenes") or []
    scenes: list[Scene] = []
    for s in raw_scenes:
        if isinstance(s, str):
            s = {"narration": s}
        narration = clean_narration(str(s.get("narration", "")))
        if not narration:
            continue
        query = _clean_tag(s.get("search_query") or "") or subject
        scenes.append(Scene(narration=narration, search_query=truncate_words(query, 60),
                            image_prompt=str(s.get("image_prompt") or query).strip(),
                            speaker=_speaker(s.get("speaker")) if multi else "A",
                            emphasis=_emphasis(s.get("emphasis"), narration), emoji=one_emoji(s.get("emoji"))))
    if not scenes:
        raise ValueError("script has no narration")
    limit = 20 if style == "chat" else 16
    while len(scenes) > limit:  # merge the shortest same-speaker neighbours to keep cuts watchable
        pairs = [k for k in range(len(scenes) - 1) if scenes[k].speaker == scenes[k + 1].speaker]
        if not pairs:
            break
        i = min(pairs, key=lambda k: len(scenes[k].narration) + len(scenes[k + 1].narration))
        a, b = scenes[i], scenes.pop(i + 1)
        a.narration = f"{a.narration} {b.narration}"
        # A merged narration must not keep only the first scene's visual plan.
        # Carry both concrete subjects forward so the footage search still represents
        # the complete spoken line rather than silently drifting out of sync.
        query_parts = []
        for part in (a.search_query, b.search_query):
            part = part.strip(" ,;")
            if part and part.casefold() not in {x.casefold() for x in query_parts}:
                query_parts.append(part)
        a.search_query = truncate_words(" ".join(query_parts), 12) or subject
        prompt_parts = [part.strip() for part in (a.image_prompt, b.image_prompt) if part and part.strip()]
        if len(prompt_parts) > 1 and prompt_parts[0].casefold() != prompt_parts[1].casefold():
            a.image_prompt = (
                f"A coherent documentary shot connecting {prompt_parts[0].rstrip('.')} "
                f"with {prompt_parts[1].rstrip('.')}. Keep both subjects visually clear; "
                "natural lighting, clear composition, vertical framing, no text."
            )
        elif prompt_parts:
            a.image_prompt = prompt_parts[0]
        a.emphasis = list(dict.fromkeys(a.emphasis + b.emphasis))[:3]
        a.emoji = a.emoji or b.emoji

    title = re.sub(r"#\w+", "", strip_emoji(str(data.get("title") or subject)))
    title = re.sub(r"[<>\"“”]", "", title).strip(" '-:")
    title = truncate_words(title or subject.title(), 90)

    hook = re.sub(r"[<>\"“”]", "", strip_emoji(str(data.get("hook_text") or ""))).strip()
    hook = truncate_words(hook, 48) if hook else truncate_words(title, 40)

    hashtags, seen = ["#shorts"], {"#shorts"}
    for h in data.get("hashtags") or []:
        h = _clean_hashtag(h)
        if h and h.lower() not in seen:
            seen.add(h.lower())
            hashtags.append(h)
    if len(hashtags) < 6:
        topic_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", str(subject) + " " + str(data.get("topic") or ""))
                       if len(w) >= 4 and w.lower() not in {"with", "from", "that", "this", "about", "your", "have", "what", "when"}]
        for word in topic_words:
            tag = "#" + word
            if tag.lower() not in seen:
                seen.add(tag.lower())
                hashtags.append(tag)
            if len(hashtags) >= 6:
                break
    for tag in ("#learnsomething", "#didyouknow", "#interestingfacts", "#explained"):
        if len(hashtags) >= 6:
            break
        if tag.lower() not in seen:
            seen.add(tag.lower())
            hashtags.append(tag)
    hashtags = hashtags[:6]

    tags, total, seen_t = [], 0, set()
    for t in list(data.get("tags") or []) + [subject, "shorts"]:
        t = truncate_words(_clean_tag(t), 60)
        cost = len(t) + (2 if " " in t else 0) + 1
        if t and t.lower() not in seen_t and total + cost <= 450:  # YouTube's tag limit is 500 chars
            seen_t.add(t.lower())
            tags.append(t)
            total += cost

    category = str(data.get("category") or "education").lower()
    if category not in CATEGORIES:
        category = "education"

    cast = [truncate_words(re.sub(r"[<>\"{}\[\]]", "", strip_emoji(str(c))).strip(), 24)
            for c in (data.get("cast") or []) if str(c).strip()]
    if multi or style in CARD_STYLES:
        defaults = DEFAULT_CAST[style]
        if style == "fantasy":
            # Preserve the LLM-created story-specific cast; never pad with generic reusable roles.
            cast = list(dict.fromkeys(cast))[:6]
        else:
            cast = (cast + defaults[len(cast):])[:2] if len(cast) < 2 else cast[:2]
        if style == "reddit":
            cast = [_handle(cast[0], "r/"), _handle(cast[1], "u/")]
    else:
        cast = []

    description = re.sub(r"(?<!\w)#\w+", "", strip_emoji(str(data.get("description") or ""))).strip()
    score = data.get("score")
    rubric = data.get("rubric") if isinstance(data.get("rubric"), dict) else {}
    return Script(
        topic=str(data.get("topic") or subject).strip(),
        title=title,
        hook_text=hook,
        scenes=scenes,
        description=description,
        hashtags=hashtags,
        tags=tags,
        category=category,
        style=style,
        language=language,
        cast=cast,
        score=int(score) if isinstance(score, (int, float)) else None,
        review=[str(x) for x in data.get("review") or []][:8],
        rubric={k: int(v) for k, v in rubric.items() if k in RUBRIC and isinstance(v, (int, float))},
    )


def word_count(text: str, language: str = "en") -> int:
    if language.split("-")[0] in ("zh", "ja"):
        return round(len(re.sub(r"\s", "", text)) / 2.2)  # ~characters per spoken "word" beat
    return len(text.split())


def write_script(llm, subject: str, settings: Settings, *, source: str = "niche",
                 avoid: list[str] | None = None, context: str = "", style: str | None = None,
                 insights: str = "", keywords: str = "", creative_brief: str = "") -> Script:
    style = style if style in STYLES else pick_style(settings, source)
    if llm is None:
        return offline_script(subject, settings, style)
    system, user = build_prompt(subject, style, settings, avoid or [], context, insights, keywords, creative_brief)
    data = llm.complete_json(system, user, SCRIPT_SCHEMA)
    try:
        script = normalize(data, subject, style, settings.language)
    except (ValueError, TypeError, AttributeError) as first_error:
        # Free OpenRouter routes can return schema-shaped JSON with an empty/missing scenes array
        # after a truncated long response. Do one small, explicit repair request instead of failing
        # the entire job or repeating the original long prompt.
        if "narration" not in str(first_error).lower() and "scene" not in str(first_error).lower():
            raise
        log.warning("Script response contained no usable narration; requesting one compact repair: %s", first_error)
        compact_system = (
            "You are a reliable YouTube Shorts scriptwriter. Return one complete JSON object only, "
            "matching the provided schema. Every scene MUST contain a non-empty spoken narration sentence. "
            "Never return an empty scenes array, blank narration, or metadata-only output."
        )
        words, n_scenes = scene_plan(style, settings.target_seconds)
        compact_user = (
            f"Topic: {subject}\nFormat: {STYLES[style]}\nLanguage: {LANGUAGES.get(settings.language, settings.language)}\n"
            f"Runtime: {settings.target_seconds} seconds; aim for about {words} spoken words across {n_scenes} scenes.\n"
            "Write a complete, coherent script: directly explain the topic early, give the cause in plain language, "
            "include a concrete example, and end with a clear payoff. Each scene must have actual narration plus "
            "a specific matching search_query and image_prompt. Keep each scene concise but meaningful. "
            "Return every required field from the schema; narration must never be blank."
        )
        data = llm.complete_json(compact_system, compact_user, SCRIPT_SCHEMA, max_tokens=4200, temperature=0.5)
        script = normalize(data, subject, style, settings.language)
    log.info("Script (%s, %d words, %d scenes): %s", style, word_count(script.narration, settings.language),
             len(script.scenes), script.title)
    return script


# --------------------------------------------------------------------------------------------------
# Script Doctor — a second pass that scores the draft for retention and fixes what is weak.
# --------------------------------------------------------------------------------------------------
_GREETING = re.compile(r"^(hi|hey|hello|welcome|what'?s up|in this video|today (we|i|i'm|we're)|let'?s talk)\b", re.I)
_CTA = re.compile(r"\b(follow|comment|subscribe|share|like|save|tell me|let me know|which one|would you)\b", re.I)
_FILLER = re.compile(r"\b(basically|actually|literally|really|very|just|kind of|sort of|you know|honestly)\b", re.I)
_NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                     r"(thir|four|fif|six|seven|eigh|nine)teen|(twen|thir|for|fif|six|seven|eigh|nine)ty|"
                     r"hundred|thousand|million|billion|trillion|half|twice|dozen|percent)\b", re.I)
_COMMON = {"about", "after", "again", "because", "before", "being", "could", "every", "first", "their", "there",
           "these", "thing", "things", "those", "which", "while", "would", "other", "still", "where", "really"}
FACTUAL_STYLES = {"facts", "listicle", "myth", "news", "explainer", "quiz"}


def lint_script(script: Script, settings: Settings) -> list[str]:
    """Flag concrete defects only; subjective preferences should not drive rewrites."""
    out: list[str] = []
    def advisory(message: str) -> None:
        # Kept as an intentional no-op: style preferences are not quality failures.
        return
    lang = script.language.split("-")[0]
    english = lang == "en"
    hook = script.scenes[0].narration if script.scenes else ""
    opening = re.split(r"(?<=[.!?。！？])\s+", hook.strip(), maxsplit=1)[0]  # the first spoken sentence
    n_hook = word_count(opening, script.language)
    if script.style not in CARD_STYLES and script.style not in MULTI_SPEAKER and n_hook > 16:
        advisory(f"The first spoken sentence is {n_hook} words; consider tightening the hook.")
    if english and _GREETING.search(hook.strip()):
        advisory("The hook opens with a greeting or 'in this video'; consider starting closer to the point.")
    target = scene_plan(script.style, settings.target_seconds)[0]
    n = word_count(script.narration, script.language)
    if target and abs(n - target) > 0.25 * target:
        advisory(f"About {n} words for a {settings.target_seconds}-second video (~{target} words fit).")
    # Scene density is part of the actual edit: avoid huge narration boxes with too few visual cuts.
    minimum_scenes, maximum_scenes = scene_count_bounds(script.style, settings.target_seconds)
    if len(script.scenes) < minimum_scenes:
        advisory(f"Only {len(script.scenes)} scene boxes for a {settings.target_seconds}-second Short; aim for {minimum_scenes}-{maximum_scenes} distinct visual beats without cutting narration.")
    elif len(script.scenes) > maximum_scenes:
        advisory(f"{len(script.scenes)} scene boxes exceed the practical {maximum_scenes}-scene target; merge only repetitive or visually redundant beats.")
    # Check the edit as a video, not just as prose. These findings feed the Script Doctor.
    if len(script.scenes) > 1:
        queries = [re.sub(r"[^a-z0-9 ]", "", sc.search_query.lower()).strip() for sc in script.scenes]
        for i in range(1, len(queries)):
            if queries[i] and queries[i] == queries[i - 1]:
                out.append(f"Scenes {i} and {i + 1} use the same footage search; change the visual when the narrated beat changes.")
                break
        for i, sc in enumerate(script.scenes, 1):
            if len(sc.image_prompt.split()) < 5:
                out.append(f"Scene {i} has an underspecified visual prompt; describe a concrete subject, action, and shot.")
                break
        if len(script.scenes) >= 4:
            short_scenes = [i for i, sc in enumerate(script.scenes, 1) if len(sc.narration.split()) <= 2]
            if len(short_scenes) >= max(3, len(script.scenes) // 2):
                out.append("Too many scenes are only a couple of words; combine beats that do not need their own visual cut.")
    # Stronger publishability checks: repeated visual plans, title/opening promise drift,
    # and endings that spend the final beat on engagement bait instead of delivery.
    if english and script.scenes and script.style in {"story", "reddit", "chat"}:
        # Deterministic continuity tripwires for common reveal failures. These are deliberately
        # narrow: they flag missing explanations for the editor rather than trying to judge all fiction.
        narration = " ".join(sc.narration for sc in script.scenes).lower()
        has_animal = bool(re.search(
            r"\b(?:kitten|cat|hamster|puppy|dog|bird|rabbit|bunny|mouse|mice|rat|pet|animal|creature)\b",
            narration,
        ))
        hidden_barrier = bool(re.search(
            r"\b(?:sealed|hidden|locked|tiny|small)\s+(?:door|hatch|panel|opening)\b"
            r"|\bbehind (?:the )?(?:wardrobe|wall|walls|panel|sealed door)\b"
            r"|\binside (?:the )?(?:wall|walls|crawlspace|wall cavity)\b"
            r"|\btrapped (?:inside|behind|within)\b",
            narration,
        ))
        access_explanation = bool(re.search(
            r"\b(?:through|via|from|into) (?:a |the )?(?:gap|hole|vent|opening|crawlspace|crack|duct|pipe|loose board|broken panel)\b"
            r"|\b(?:gap|hole|vent|opening|crack|duct|pipe|loose board|broken panel) (?:let|allowed|gave|provided|connected|led) (?:it|the animal|the kitten|the cat|the creature|a way)\b"
            r"|\b(?:squeezed through|slipped through|crawled through|climbed through|fell through|came through|got inside|got in|entered through|wandered in|left open|from outside)\b",
            narration,
        ))
        explicit_magic = bool(re.search(
            r"\b(?:magic|magical|supernatural|ghost|haunted|curse|enchanted|teleported|impossible)\b",
            narration,
        ))
        if has_animal and hidden_barrier and not access_explanation and not explicit_magic:
            out.append(
                "Fiction continuity: an animal is found behind a sealed/hidden barrier, but the narration "
                "never explains how it entered. Plant a believable route in (gap, hole, vent, opening), "
                "or establish a consistent supernatural rule."
            )
        has_knocking = bool(re.search(r"\b(?:knock|knocks|knocked|knocking|rattle|rattled|taps|tapped)\b", narration))
        physical_sound_cause = bool(re.search(
            r"\b(?:struck|hit|banged|thumped|kicked) (?:the )?(?:door|panel|wood(?:en)?|frame|wall|board)\b"
            r"|\b(?:pawed|scratched|clawed|bumped|thumped) (?:at|against|on) (?:the )?(?:door|panel|wood(?:en)?|frame|wall|board)\b"
            r"|\b(?:body|tail|paw|claw) (?:struck|hit|banged|thumped|bumped) (?:the )?(?:door|panel|wood(?:en)?|frame|wall|board)\b"
            r"|\b(?:rattled|shook|banged against) (?:the )?(?:door|panel|wood(?:en)?|frame|wall|board)\b",
            narration,
        ))
        if has_animal and has_knocking and not physical_sound_cause and not explicit_magic:
            out.append(
                "Fiction continuity: the knocking is not physically connected to the animal's actions. "
                "Show what strikes or moves against the door/panel, or establish a consistent supernatural rule."
            )
        long_confinement = bool(re.search(
            r"\b(?:all summer|for (?:several )?(?:weeks|months|years)|for the past (?:weeks|months|year)|"
            r"since (?:january|february|march|april|may|june|july|august|september|october|november|december))\b",
            narration,
        ))
        supply_explanation = bool(re.search(
            r"\b(?:food|water|drank|ate|fed|feeding|rainwater|drain|pipe|supplies|"
            r"kept it alive|survived because|survived on|could reach)\b",
            narration,
        ))
        if has_animal and hidden_barrier and long_confinement and not supply_explanation and not explicit_magic:
            out.append(
                "Fiction continuity: an animal appears to survive a long confinement, but the narration "
                "does not explain access to food/water or another credible survival mechanism. Shorten the "
                "duration or establish the mechanism before the reveal."
            )
    if english and script.scenes:
        def content_words(value: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", value.lower())
                    if len(w) > 3 and w not in _COMMON and w not in _STOP}
        title_words = content_words(script.title)
        opening_words = content_words(script.scenes[0].narration)
        if title_words and opening_words:
            overlap = len(title_words & opening_words) / max(1, min(len(title_words), len(opening_words)))
            if overlap < 0.20 and len(title_words) >= 3:
                out.append("Title and spoken opening may promise different subjects; align the title with what the first line actually delivers.")
        normalized_queries = [
            re.sub(r"[^a-z0-9 ]", "", sc.search_query.lower()).strip()
            for sc in script.scenes
        ]
        seen_queries: dict[str, int] = {}
        for i, query in enumerate(normalized_queries, 1):
            if not query:
                out.append(f"Scene {i} has no usable visual search query; provide a concrete subject that can be found or illustrated.")
                break
            if query in seen_queries and len(script.scenes) >= 3:
                out.append(f"Scenes {seen_queries[query]} and {i} reuse the same visual search; vary the shot when the narrated beat changes.")
                break
            seen_queries[query] = i
        last_line = script.scenes[-1].narration.strip()
        if _CTA.search(last_line) and len(last_line.split()) <= 12:
            advisory("The final beat leans on a CTA; consider ending on the answer or payoff.")
        if script.style in {"story", "reddit", "chat"} and re.search(r"\bjust in case\b|\bfor some reason\b", last_line, re.I):
            out.append("The final line may leave a vague dangling implication (for example, 'just in case'); state what the character fears and connect it to the resolved story, or end on the actual consequence.")
        if script.style in {"story", "reddit", "chat"} and len(script.scenes) >= 3:
            ending_words = content_words(" ".join(sc.narration for sc in script.scenes[-2:]))
            hook_words = content_words(script.scenes[0].narration)
            if hook_words and len(ending_words & hook_words) == 0:
                out.append("The ending may not reconnect to the opening; make the final two scenes resolve the central question or explain the reveal's consequence.")
        for i, sc in enumerate(script.scenes, 1):
            if not sc.search_query.strip() or len(sc.search_query.split()) < 2:
                out.append(f"Scene {i} has a vague visual search; use 2–5 concrete English words matching the narrated subject.")
                break
        for i, sc in enumerate(script.scenes, 1):
            if not sc.image_prompt.strip() or len(sc.image_prompt.split()) < 7:
                out.append(f"Scene {i} needs a specific visual prompt with subject, visible action, setting, and framing.")
                break
    if english:
        repeated_lines = [re.sub(r"[^a-z0-9 ]", "", sc.narration.lower()).strip() for sc in script.scenes]
        for i in range(1, len(repeated_lines)):
            if repeated_lines[i] and repeated_lines[i] == repeated_lines[i - 1]:
                out.append(f"Scenes {i} and {i + 1} repeat the same narration; remove the duplicate.")
                break

        # Catch morphology-based word echoes inside a sentence ("a bustling city ... bustles")
        # without flagging ordinary repeated nouns across separate story beats.
        def echo_root(token: str) -> str:
            if len(token) > 6 and token.endswith("ing"):
                return token[:-3]
            if len(token) > 5 and token.endswith("es"):
                return token[:-2]
            if len(token) > 4 and token.endswith("s"):
                return token[:-1]
            if len(token) > 5 and token.endswith("ed"):
                return token[:-2]
            return token

        for scene_no, sc in enumerate(script.scenes, 1):
            sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", sc.narration) if x.strip()]
            found_echo = False
            for sentence in sentences:
                words = re.findall(r"[a-z]+", sentence.lower())
                roots = [(i, w, echo_root(w)) for i, w in enumerate(words)]
                for idx, word, root in roots:
                    if len(root) < 4 or root in _STOP or word == root:
                        continue
                    for _other_idx, other_word, other_root in roots[idx + 1:idx + 8]:
                        if root == other_root and word != other_word:
                            out.append(
                                f"Scene {scene_no} has a word-echo ('{word}' and '{other_word}') that repeats the same idea; rewrite the sentence."
                            )
                            found_echo = True
                            break
                    if found_echo:
                        break
                if found_echo:
                    break
            if found_echo:
                break
    # A story payoff is more important than a call to action; CTAs are optional.
    if lang not in ("zh", "ja"):
        for k, sc in enumerate(script.scenes, 1):
            sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", sc.narration) if x.strip()]
            longest = max((len(x.split()) for x in sentences), default=0)
            if longest > 22:
                out.append(f"Scene {k} has a {longest}-word sentence; shorten it for natural spoken delivery.")
                break
            if len(sc.narration.split()) > 34:
                out.append(f"Scene {k} has {len(sc.narration.split())} words; break the narration into shorter spoken beats.")
                break
    if english:
        topic = {w.lower() for w in re.findall(r"[A-Za-z]+", script.topic)}
        counts: dict[str, int] = {}
        for w in re.findall(r"[A-Za-z']+", script.narration.lower()):
            if len(w) > 4 and w not in _COMMON and w not in topic:
                counts[w] = counts.get(w, 0) + 1
        word, times = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
        if times >= 4:
            out.append(f"'{word}' is said {times} times; vary the wording.")
        fillers = len(_FILLER.findall(script.narration))
        if fillers > 3:
            advisory(f"{fillers} filler words (basically, really, just...) slow the pace.")
        if script.style in FACTUAL_STYLES and not _NUMBER.search(script.narration):
            advisory("No concrete number anywhere; a precise figure can help, but is not mandatory.")
    # Only actionable defects should trigger a Script Doctor rewrite. The other findings
    # above are useful editing suggestions, not reasons to hold up a generated video.
    hard_prefixes = (
        "Fiction continuity:",
        "Scene ",  # missing/ unusable visual query or prompt
        "Scenes ",  # duplicate narration
        "The ending may not reconnect to the opening",
        "The final line may leave a vague dangling implication",
    )
    return [
        issue for issue in out
        if issue.startswith(hard_prefixes)
        and (
            issue.startswith("Fiction continuity:")
            or "no usable visual search query" in issue
            or "vague visual search" in issue
            or "specific visual prompt" in issue
            or "repeat the same narration" in issue
            or "ending may not reconnect" in issue
            or "final line may leave" in issue
        )
    ]


def rubric_score(rubric: dict | None) -> int | None:
    """Weighted 0-100 retention score from 0-10 sub-scores (missing parts count as average)."""
    if not isinstance(rubric, dict):
        return None
    total, seen = 0.0, 0
    for key, (weight, _) in RUBRIC.items():
        try:
            v = max(0.0, min(10.0, float(rubric[key])))
            seen += 1
        except (KeyError, TypeError, ValueError):
            v = 5.0
        total += weight * v
    return round(total * 10) if seen else None


def _score_of(data: dict) -> tuple[int | None, dict[str, int]]:
    rubric = data.get("rubric")
    score = rubric_score(rubric)
    clean = ({k: max(0, min(10, int(v))) for k, v in rubric.items() if k in RUBRIC and isinstance(v, (int, float))}
             if isinstance(rubric, dict) else {})
    if score is None:  # a plain overall score (older prompts, simple models)
        try:
            score = max(0, min(100, int(data.get("score"))))
        except (TypeError, ValueError):
            score = None
    return score, clean


def review_prompt(script: Script, settings: Settings, rewrite: bool, checks: list[str] | None = None) -> tuple[str, str]:
    words, n_scenes = scene_plan(script.style, settings.target_seconds)
    lang = LANGUAGES.get(script.language, script.language)
    draft = {k: v for k, v in script.to_dict().items() if k not in ("style", "language", "score", "review", "rubric")}
    system = ("You are a ruthless YouTube Shorts retention editor. You know that viewers decide in 1-2 "
              "seconds and swipe away at any slow moment. You reply with a single JSON object and nothing else.")
    rubric = "\n".join(f"- {k}: {why}" for k, (_, why) in RUBRIC.items())
    found = ("Automatic checks already found:\n" + "\n".join(f"- {c}" for c in checks) + "\n") if checks else ""
    task = (f"""Then rewrite it into the strongest version you can. Keep the same subject, format, language ({lang}),
the same JSON structure and a narration length appropriate for the selected runtime (target about {words} words). Aim for {n_scenes} scenes, within {scene_count_bounds(script.style, settings.target_seconds)[0]}–{scene_count_bounds(script.style, settings.target_seconds)[1]}; preserve all useful narration while redistributing it into more distinct, visual beats when needed. Fix every issue and automatic finding. Do a complete rewrite where needed—not a cosmetic line edit. Rebuild weak structure, replace generic phrasing, strengthen cause-and-effect, improve spoken rhythm, ensure visuals match each beat, and make the payoff earn the opening. A call to action is optional and must never replace the final story beat. Put the rewritten script in "script".
Creative-quality requirements:
- Do not default to "Did you know?" or any repeated stock opening. Keep the opening specific to this topic and distinct from recent titles/context.
- Read the narration as one complete spoken script before judging individual scenes. It must flow naturally from the first line to the final payoff, with no disconnected lines or abrupt jumps—even in facts/list formats.
- Line-edit the whole narration: replace vague or awkward wording, remove repeated information, clarify references, improve transitions, and vary sentence rhythm. Reject writing that merely sounds dramatic but says little.
- Avoid canned AI phrases and trailer-style overacting. Keep the voice engaging but believable; use punchy lines selectively and give explanations enough room to make sense.
- MANDATORY READ-ALOUD EDIT: mentally read the complete narration at normal speaking speed. Rewrite any sentence that sounds like an essay, caption, generic AI narration, slogan, or a collection of dramatic fragments. The final script should sound like a real person telling this exact story to one interested viewer, with natural contractions, varied sentence lengths, smooth transitions, and no fake profundity.
- Cut vague emotional labels and replace them with specific reactions or consequences. Remove generic scaffolding, repeated explanations, empty suspense, and a second twist that has not been earned. Every important object or detail should pay off, and the viewer must understand the central event after one listen.
- Run a sentence-level redundancy audit: catch repeated word roots and meaning echoes (for example, "a bustling city ... bustles", "whispered quietly", or "tiny miniature"). Rewrite tautologies rather than merely deleting one adjective at random.
- For fiction, audit the premise's rules and causal chain: what makes the reveal true, what evidence plants it, what exact danger or need exists, what the character fears and why, why the protagonist's choice matters, and what changes because of it? Check physical scale, access, object placement, elapsed time, and how the reveal relates to the opening clue. HARD PLAUSIBILITY CHECK: test whether each character or animal could survive the stated time and conditions, including food, water, air, temperature, shelter, injury, age, and physical ability. Do not use vague explanations like 'it survived on crumbs' to excuse a biologically impossible duration. Either provide a credible cause supported by the story, establish an explicit supernatural rule with earlier clues, or change the duration/condition. A trapped creature must have a clear route into the space and a believable way to survive; the narrator must be able to reach it, and freeing it must resolve the original problem. A future-self reveal or warning must explain its connection and consequence in spoken narration, not leave the viewer to invent the missing logic.
- Reject generic stock twists and familiar horror imagery unless the particular setup earns them. The story should be recognizably about this exact premise, not a template with swapped nouns.
- After editing, join all scene narration into one continuous passage and check chronology, point of view, motivations, transitions, and repeated information. Split it back into scenes only at natural spoken/visual beats; never make the prose sound chopped up to fit cards.
- Verify that title, on-screen hook, and spoken opening promise the same story. Reject a mismatched clickbait title or an unrelated visual gimmick.
- Each scene must have a concrete, searchable visual directly matching its narration. Avoid vague/repeated adjacent search queries; vary the actual subject or shot when the narration changes.
- Do not claim a visual is secured just because a search query exists; use specific search_query and a concrete image_prompt that describes a plausible shot.
- Write for the ear, not an article: use contractions and natural spoken phrasing where appropriate, varied sentence lengths, and one clear idea per breath.
- Apply these narration standards to every topic and every format, not only science or factual videos: natural human speech, familiar words, clear logic, useful context, and no robotic or article-like phrasing.
- Explain unfamiliar terms, acronyms, people, places, events, and concepts briefly in plain language when they matter; never leave jargon unexplained or pile up names.
- EXPLAIN THE TOPIC, DON'T JUST TEASE IT (mandatory for facts, explainer, science, news, myth, quiz, listicle, and how-to): within the first 1–2 spoken sentences, explicitly name what the topic is and give a plain-language definition or direct answer. Do not make viewers infer the subject from suspense, examples, dramatic claims, or visuals. Then explain the mechanism/cause in everyday language, give one concrete example or consequence, and close the loop on the opening question. Every unfamiliar name or event must get enough context to understand why it matters. A viewer who has never heard of the topic should be able to explain the basic idea after one listen. If the runtime is short, simplify the topic instead of omitting the explanation.
- Choose an opening that fits the topic instead of repeating a template: a direct question when appropriate, otherwise a concrete detail, conflict, moment, or surprising claim.
- Ensure each line develops the previous one. Explain why details matter, use examples or comparisons when helpful, and cut empty suspense, filler, repeated ideas, generic claims, and forced drama.
- Match narration to the selected format: stories need believable progression, facts need clear explanations, quizzes need a clear answer, tutorials need usable steps, news needs context, dialogue needs believable voices, and motivational scripts need substance rather than slogans.
- Keep every claim accurate and appropriately qualified; simple language must not mean misleading oversimplification.
- HUMAN NARRATOR VOICE CHECK: make this sound like a smart, naturally conversational creator speaking to one viewer. Where it fits, use an occasional light, relatable joke or observation; do not force humor into every beat or undermine serious/scary/emotional moments. For a question or hypothetical, consider stating the actual question first, voicing the viewer's likely intuitive assumption briefly, then revealing the accurate answer and explaining why. This is a flexible pattern, not a mandatory template.
- FACT EXPLANATION CHECK: for science, facts, explainers, quizzes, and hypotheticals, ensure the narration gives a clear causal chain in ordinary language: what happens, why it happens, and what the viewer would experience. Use a small analogy or relatable comparison when it improves understanding. Keep real physics and uncertainty accurate; do not sensationalize or turn possibilities into certainties.
- STORYTELLING VOICE CHECK: stories should sound like someone recounting real events, not an outline being read aloud. Preserve a clear point of view, natural chronology, concrete actions, reactions, and smooth transitions. First person is appropriate only when the story's chosen narrator is the person experiencing it. Use humor only when character- and moment-appropriate; never bolt a joke onto a tense or emotional beat.
- VARIETY CHECK: don't make every script follow the same question -> joke -> truth formula. Keep the channel voice recognizable through clarity, warmth, curiosity, and occasional wit, while varying openings and pacing to suit each topic.
- For factual, science, explainer, quiz and hypothetical videos, begin by asking the actual question in natural speech when that fits the topic (for example, "So, what would happen if you stayed awake for 24 hours?"). Do not bury the question under a dramatic but vague hook.
- Prefer common words over specialist jargon. Whenever an unfamiliar technical term matters, explain immediately in plain language what it means and what it does. Never use a technical name as if the audience already knows it.
- Explain cause and effect briefly: what is happening, why it happens, and what the viewer would notice. Keep the explanation conversational and easy to follow aloud.
- Avoid long sentences with multiple clauses. Prefer sentences around 8-18 words; split a sentence when a listener would need to mentally rewind.
- Use punctuation to support natural TTS pauses, but do not add ellipses or commas everywhere. Do not include stage directions in narration.
- Keep transitions between scenes logical: each line should connect to the previous thought or set up the next, not feel like disconnected facts.
- Do not pad the target length with filler. If a script is short, add a meaningful example, mechanism, or consequence instead.
- For story-driven scripts, require a complete mini-arc even at short runtimes: establish a vivid situation, introduce a specific disturbance, show a reaction and consequential choice, deliver a planted reveal, then show its immediate meaning or consequence. These are narrative functions, not a required scene count; one scene may carry more than one function when clear. The ending must connect the reveal to the opening and answer at least one important question. Never end on unexplained dialogue, a character waving, or an eerie image alone. If both reveal and payoff cannot fit, simplify the premise rather than cutting the payoff.
- Before returning a fiction script, silently run a FINAL-PAYOFF TEST: (a) what question did the opening make the viewer ask? (b) what does the reveal actually mean? (c) what does the protagonist do or feel because of it? (d) does the last line complete the emotional/causal arc, or vaguely hint at another event with phrases like "just in case"? Remove a dangling hint unless the story clearly explains what is feared and why. If the ending only introduces another question, rewrite it. The final two scenes together must contain the reveal AND its consequence. Do not rely on the title, visuals, or audience imagination to explain a crucial relationship; the spoken narration must make it clear.
- Treat every video as self-contained. Do not write a serial episode, cliffhanger designed only to force a sequel, or recurring-character continuity unless explicitly requested. Do not append "What do you think?" by habit; prioritize a natural final line that pays off the story. CTA is optional.
- Do not imitate a named creator's exact wording, characters, plot, or signature phrases. Learn only broad craft principles: visual specificity, clear cause-and-effect, emotional reactions, and satisfying standalone payoffs.
Format rules to keep:
{_format_rules(script.style, settings.target_seconds)}""" if rewrite else 'Return only "rubric" and "issues".')
    user = f"""Review this YouTube Short script ({STYLES[script.style]}).

{json.dumps(draft, ensure_ascii=False, indent=1)}

Grade it in "rubric" with a whole number from 0 to 10 for each of:
{rubric}
Be strict and calibrated: 5 is an average Short, 8 is better than most viral Shorts, 10 is nearly impossible.
List the concrete problems in "issues" (max 5, short sentences, most important first).
{found}{task}"""
    return system, user


def _grade(llm, script: Script, settings: Settings) -> tuple[int | None, dict[str, int], list[str]]:
    system, user = review_prompt(script, settings, rewrite=False, checks=lint_script(script, settings))
    data = llm.complete_json(system, user, SCORE_SCHEMA, max_tokens=2000, temperature=REVIEW_TEMPERATURE)
    score, rubric = _score_of(data)
    return score, rubric, [str(i).strip() for i in data.get("issues") or [] if str(i).strip()][:5]


def _enforce_story_payoff(llm, script: Script, settings: Settings) -> tuple[Script, int | None, dict[str, int], list[str]]:
    """Run a dedicated ending check for narrative scripts; rewrite once if the payoff is weak."""
    try:
        score, rubric, issues = _grade(llm, script, settings)
    except Exception as e:
        log.warning("Story payoff check skipped: %s", e)
        return script, script.score, script.rubric, list(script.review or [])

    quality_dimensions = ("payoff", "specificity", "pacing", "curiosity")
    weak_dimensions = [key for key in quality_dimensions if rubric.get(key, 5) < 7]
    if script.style not in {"story", "reddit", "chat"} or not weak_dimensions:
        return script, score, rubric, issues

    forced_checks = lint_script(script, settings) + [
        "STORY PAYOFF FAILED: rewrite the final two scenes so the reveal answers a question raised by the opening and shows a concrete consequence. Do not end on a new unexplained mystery."
    ]
    system, user = review_prompt(script, settings, rewrite=True, checks=forced_checks)
    try:
        data = llm.complete_json(system, user, REVIEW_SCHEMA, max_tokens=5000, temperature=REVIEW_TEMPERATURE)
        if not isinstance(data.get("script"), dict):
            return script, score, rubric, issues
        candidate = normalize(data["script"], script.topic, script.style, script.language)
        target = scene_plan(script.style, settings.target_seconds)[0]
        count = word_count(candidate.narration, script.language)
        if not (0.75 * target <= count <= 1.25 * target):
            log.warning("Story payoff rewrite ignored: %d words for a %d-word target", count, target)
            return script, score, rubric, issues
        new_score, new_rubric, new_issues = _grade(llm, candidate, settings)
        old_weakness = sum(rubric.get(key, 5) for key in quality_dimensions)
        new_weakness = sum(new_rubric.get(key, 5) for key in quality_dimensions)
        if new_weakness > old_weakness or (
            new_weakness == old_weakness and (new_score or 0) >= (score or 0)
        ):
            log.info("Story quality check: accepted rewrite (weak dimensions total %s -> %s)",
                     old_weakness, new_weakness)
            return candidate, new_score, new_rubric, new_issues
        log.info("Story payoff check: rewrite did not improve payoff; kept prior script")
    except Exception as e:
        log.warning("Story payoff rewrite skipped: %s", e)
    return script, score, rubric, issues


def review_script(llm, script: Script, settings: Settings) -> Script:
    """Score the script and (in rewrite mode) replace it with an improved version. The rewrite is graded
    again and only kept when it scores at least as well as the draft, so the stored score belongs to the
    script that is actually used (if that second grading fails, the rewrite is used with the draft's score).
    With SCRIPT_REVIEW_ROUNDS above 1 a script still below SCRIPT_TARGET_SCORE is reviewed again. Never
    fails the video: any unusable answer keeps what we have."""
    mode = settings.script_review
    if llm is None or mode == "off":
        return script
    if mode == "score":
        script, score, rubric, issues = _enforce_story_payoff(llm, script, settings)
        if score is None and not rubric:
            return script
        script.score, script.rubric, script.review = score, rubric, _with_checks(issues, script, settings)
        log.info("Script Doctor: %s/100%s", score if score is not None else "?", f" — {issues[0]}" if issues else "")
        return script

    best, best_score, best_rubric, best_issues = script, None, {}, []
    target = scene_plan(script.style, settings.target_seconds)[0]
    for round_no in range(1, max(1, settings.script_review_rounds) + 1):
        system, user = review_prompt(best, settings, rewrite=True, checks=lint_script(best, settings))
        try:
            data = llm.complete_json(system, user, REVIEW_SCHEMA, max_tokens=5000, temperature=REVIEW_TEMPERATURE)
        except Exception as e:
            log.warning("Script Doctor skipped: %s", e)
            break
        score, rubric = _score_of(data)
        issues = [str(i).strip() for i in data.get("issues") or [] if str(i).strip()][:5]
        if round_no == 1 or best_score is None:
            best_score, best_rubric, best_issues = score, rubric, issues
        better = None
        if isinstance(data.get("script"), dict):
            try:
                cand = normalize(data["script"], script.topic, script.style, script.language)
                n = word_count(cand.narration, script.language)
                if 0.75 * target <= n <= 1.25 * target:
                    better = cand
                else:
                    log.warning("Script Doctor rewrite ignored: %d words for a %d-word target", n, target)
            except ValueError as e:
                log.warning("Script Doctor rewrite ignored: %s", e)
        if better is None:
            break
        try:
            new_score, new_rubric, new_issues = _grade(llm, better, settings)
        except Exception as e:  # can't verify: trust the editor's rewrite, keep the draft's score
            log.warning("Script Doctor could not grade its rewrite (%s); using it", e)
            best = better
            break
        # A verified fix to a concrete story-logic gap outranks a tiny rubric-score drop.
        # Otherwise the score guard could preserve a more polished but causally broken draft.
        continuity_before = [x for x in lint_script(best, settings) if x.startswith("Fiction continuity:")]
        continuity_after = [x for x in lint_script(better, settings) if x.startswith("Fiction continuity:")]
        continuity_improved = len(continuity_after) < len(continuity_before)
        if new_score is not None and best_score is not None and new_score < best_score and not continuity_improved:
            log.info("Script Doctor rewrite scored %d, below the draft's %d: kept the draft", new_score, best_score)
            break
        best, best_score = better, new_score if new_score is not None else best_score
        best_rubric, best_issues = new_rubric, new_issues  # never show the draft's sub-scores for the rewrite
        remaining_logic_gaps = any(
            x.startswith("Fiction continuity:") for x in lint_script(best, settings)
        )
        if best_score is not None and best_score >= settings.script_target_score and not remaining_logic_gaps:
            break
    # Preserve the best verified result before the optional narrative-payoff pass. If that
    # extra provider call fails, the already-graded rewrite and its score must remain intact.
    best.score, best.rubric, best.review = best_score, best_rubric, best_issues
    if best.style in {"story", "reddit", "chat"}:
        best, best_score, best_rubric, best_issues = _enforce_story_payoff(llm, best, settings)
    best.score, best.rubric = best_score, best_rubric
    best.review = _with_checks(best_issues, best, settings)
    changed = "rewritten" if best is not script else "kept"
    log.info("Script Doctor: %s/100, %s%s", best_score if best_score is not None else "?", changed,
             f" — {best.review[0]}" if best.review else "")
    return best


def _with_checks(issues: list[str], script: Script, settings: Settings) -> list[str]:
    """The editor's remaining notes, then up to three automatic findings it didn't already mention."""
    out = list(issues)
    for c in lint_script(script, settings)[:3]:
        if len(out) >= 8:
            break
        out.append(f"Check: {c}")
    return out


# --------------------------------------------------------------------------------------------------
# Translation — one idea, many languages, same footage.
# --------------------------------------------------------------------------------------------------
def translate_script(llm, script: Script, language: str) -> Script:
    lang = LANGUAGES.get(language, language)
    src = {k: v for k, v in script.to_dict().items() if k not in ("style", "language", "score", "review", "rubric")}
    system = ("You are a native-level translator and YouTube Shorts adapter. You reply with a single JSON "
              "object and nothing else.")
    user = f"""Adapt this YouTube Short into {lang}.

{json.dumps(src, ensure_ascii=False, indent=1)}

Rules:
- Translate title, hook_text, cast, description and every scene's narration into natural spoken {lang},
  the way a native creator would say it (not word for word). Keep numbers and names accurate.
- Keep EXACTLY {len(script.scenes)} scenes in the same order, with the same speaker for each.
- Keep search_query and image_prompt unchanged (English).
- emphasis: the same key words in {lang}, copied exactly from your translated narration.
- Hashtags and tags in {lang} where people search in {lang}, otherwise English.
Return the same JSON structure."""
    data = llm.complete_json(system, user, SCRIPT_SCHEMA, temperature=0.4)  # faithful, not creative
    out = normalize(data, script.topic, script.style, language)
    if len(out.scenes) != len(script.scenes):
        raise ValueError(f"translation to {language} has {len(out.scenes)} scenes, expected {len(script.scenes)}")
    for mine, theirs in zip(out.scenes, script.scenes):
        mine.search_query, mine.image_prompt, mine.speaker = theirs.search_query, theirs.image_prompt, theirs.speaker
    out.category = script.category
    if script.style in CARD_STYLES:
        out.cast = list(script.cast)  # r/community and u/name are handles, not words to translate
    return out


LOCALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"language": {"type": "string"}, "title": {"type": "string"},
                               "description": {"type": "string"}},
                "required": ["language", "title", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


def localize_metadata(llm, script: Script, languages: list[str]) -> dict[str, dict[str, str]]:
    """The title and description in other languages, for YouTube's per-language metadata: viewers whose
    YouTube is set to one of them see the video under a title in their own language. One LLM call."""
    langs = [x for x in dict.fromkeys(languages) if x and x != script.language]
    if not langs:
        return {}
    names = ", ".join(f"{x} ({LANGUAGES.get(x.split('-')[0], x)})" for x in langs)
    system = ("You are a native-level translator for YouTube creators. You reply with a single JSON object "
              "and nothing else.")
    user = f"""Translate this YouTube Short's title and description into each of these languages: {names}.

Title: {script.title}
Description: {script.description}

Write them the way a native creator in that language would (not word for word), and keep names and numbers
exact. Title: max 90 characters, no hashtags, no quotes. Description: the same sentences, no hashtags.
Return JSON: {{"items": [{{"language": code, "title": ..., "description": ...}}, ...]}} with the codes above."""
    data = llm.complete_json(system, user, LOCALIZE_SCHEMA, max_tokens=4000, temperature=0.3)
    out: dict[str, dict[str, str]] = {}
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("language") or "").strip()
        title = re.sub(r"#\w+", "", strip_emoji(str(item.get("title") or ""))).strip(" '\"-:")
        if code in langs and title:
            desc = re.sub(r"(?<!\w)#\w+", "", strip_emoji(str(item.get("description") or ""))).strip()
            out[code] = {"title": truncate_words(title, 90), "description": desc}
    return out


def load_script(path: str | Path, settings: Settings) -> Script:
    """Read a script you wrote or edited yourself (e.g. the ``script.json`` of an earlier video) and
    clean it up exactly like an AI-written one, so it can be rendered without calling an LLM."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object with 'title' and 'scenes'")
    return script_from_data(data, settings, fallback_subject=Path(path).stem)


def script_from_data(data: dict, settings: Settings, fallback_subject: str = "my video") -> Script:
    subject = str(data.get("topic") or data.get("title") or fallback_subject)
    style = data.get("style") if data.get("style") in STYLES else "facts"
    if style not in MULTI_SPEAKER and any(_speaker(s.get("speaker")) == "B" for s in data.get("scenes") or []
                                          if isinstance(s, dict)):
        style = "dialogue"
    language = str(data.get("language") or settings.language)
    return normalize(data, subject, style, language)


# --------------------------------------------------------------------------------------------------
# Offline writer — lets `demo` produce a real video with no API keys at all.
# --------------------------------------------------------------------------------------------------
_DEMO = {
    "title": "Octopuses Are Basically Aliens",
    "hook_text": "THIS ANIMAL HAS 3 HEARTS",
    "scenes": [
        ("An octopus has three hearts, and one of them stops when it swims.", "octopus swimming",
         ["three hearts"], "🐙"),
        ("Its blood is blue, because it carries oxygen with copper instead of iron.", "blue ocean water",
         ["blue", "copper"], "🩸"),
        ("Most of its neurons are not in its head. They are spread through its eight arms.", "octopus tentacles",
         ["neurons", "eight"], "🧠"),
        ("Each arm can taste what it touches, using thousands of sensors in its suckers.", "octopus suckers closeup",
         ["taste"], "👅"),
        ("And it changes color in a split second to vanish against the reef.", "coral reef fish", ["vanish"],
         "🫥"),
        ("Which fact surprised you most? Follow for more.", "deep sea diver", [], "🤯"),
    ],
    "description": "Three hearts, blue blood and arms that can taste. The octopus might be the strangest "
                   "animal on Earth. Which fact surprised you most?",
    "hashtags": ["#octopus", "#ocean", "#animals", "#facts"],
    "tags": ["octopus facts", "ocean animals", "marine biology", "weird animals", "animal facts"],
    "category": "education",
}

_DEMO_DIALOGUE = [
    ("A", "Wait, octopuses have three hearts?", "octopus swimming"),
    ("B", "Three. And one of them stops beating every time it swims.", "octopus underwater"),
    ("A", "That can't be healthy.", "coral reef"),
    ("B", "It's why they'd rather crawl. Swimming wears them out.", "octopus crawling"),
    ("A", "Okay, but why is their blood blue?", "blue ocean water"),
    ("B", "Copper. Their blood carries oxygen with copper instead of iron.", "deep sea"),
    ("A", "So they're basically aliens.", "jellyfish glowing"),
    ("B", "Pretty much. Follow for more ocean weirdness.", "ocean waves"),
]

_DEMO_REDDIT = [
    ("Am I wrong for feeding my neighbor's cat every night for a whole year?", "orange cat window"),
    ("Every evening at seven, this orange cat sat outside my kitchen door and cried like it hadn't eaten in days.",
     "cat door"),
    ("So I bought him food. Good food. It became our little routine.", "cat eating"),
    ("Last week my neighbor knocked and asked why her cat was getting so fat.", "front door"),
    ("Turns out she feeds him at six. The family across the street feeds him at eight.", "suburban street"),
    ("He has been eating three dinners a night, at three different houses.", "fat cat sleeping"),
    ("We made a group chat. He is on a diet now, and he is furious about it.", "grumpy cat"),
    ("Would you have kept feeding him? Follow for the update.", "cat sunset"),
]

_DEMO_CHAT = [
    ("A", "are you still at the aquarium", "aquarium tunnel"),
    ("B", "yeah why", "aquarium fish"),
    ("A", "the octopus tank. look at it", "octopus tank"),
    ("B", "it's empty??", "empty aquarium"),
    ("A", "it's not empty. look closer", "coral reef"),
    ("B", "omg it's on the glass. it was the same color as the rock", "octopus camouflage"),
    ("A", "they can change color in under a second", "octopus color"),
    ("B", "that's actually terrifying", "dark ocean"),
    ("A", "wait till you hear it has three hearts", "octopus swimming"),
    ("B", "ok follow for part 2", "ocean waves"),
]


def offline_script(subject: str, settings: Settings, style: str = "facts") -> Script:
    s = (subject or "").strip()
    if style in MULTI_SPEAKER:
        lines = _DEMO_DIALOGUE if style == "dialogue" else _DEMO_CHAT
        data = dict(_DEMO, topic="octopus superpowers", cast=DEFAULT_CAST[style] if style == "dialogue"
                    else ["Maya", "Me"],
                    scenes=[{"speaker": sp, "narration": n, "search_query": q, "image_prompt": q}
                            for sp, n, q in lines])
        if style == "chat":
            data.update(title="She Was Staring At An Empty Tank", hook_text="IT WAS NEVER EMPTY")
        else:
            data.update(title="Wait, Octopuses Have Three Hearts?", hook_text="3 HEARTS?!")
        return normalize(data, "octopus superpowers", style, "en")
    if style == "reddit":
        data = dict(_DEMO, topic="a cat with three families", title="He Was Eating Three Dinners A Night",
                    hook_text="THREE DINNERS A NIGHT", cast=["r/confessions", "u/quiet_otter22"],
                    description="My neighbor's cat had a secret. Would you have kept feeding him?",
                    hashtags=["#redditstories", "#cats", "#storytime"],
                    tags=["reddit stories", "cat story", "funny cat", "storytime", "confession"],
                    category="entertainment",
                    scenes=[{"narration": n, "search_query": q, "image_prompt": q} for n, q in _DEMO_REDDIT])
        return normalize(data, "a cat with three families", "reddit", "en")
    if not s or "octopus" in s.lower() or s.lower() in {"demo", "random"}:
        d = _DEMO
        data = dict(d, topic="octopus superpowers",
                    scenes=[{"narration": n, "search_query": q, "image_prompt": q, "emphasis": e, "emoji": m}
                            for n, q, e, m in d["scenes"]])
        return normalize(data, "octopus superpowers", "facts", "en")
    lines = [
        f"Most people think they know {s}. They don't.",
        f"The story of {s} is stranger than it looks.",
        "The details surprised even the experts who studied it.",
        "And one small fact changes how you see the whole thing.",
        "Would you have guessed that? Follow for more.",
    ]
    data = {
        "topic": s, "title": f"The Truth About {s.title()}", "hook_text": "YOU WON'T EXPECT THIS",
        "scenes": [{"narration": n, "search_query": s, "image_prompt": s} for n in lines],
        "description": f"A quick look at {s}. What would you add?",
        "hashtags": ["#facts"], "tags": [s, "facts"], "category": "education",
    }
    return normalize(data, s, style if style not in MULTI_SPEAKER else "facts", settings.language)
