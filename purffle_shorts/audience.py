"""Conservative audience classification for YouTube's Made for Kids declaration.

Only mark a video as child-directed when the content contains strong explicit signals.
A topic that children may enjoy (such as general facts or fantasy) is not sufficient.
Ambiguous cases preserve the creator's configured MADE_FOR_KIDS setting.
"""
from __future__ import annotations

import re
from typing import Any

_CHILD_DIRECTED = re.compile(
    r"\b(?:made\s+for\s+kids|for\s+(?:young\s+)?children|for\s+kids|"
    r"kids?\s+(?:learning|education|educational|science|facts|stories)|"
    r"children['’]?s\s+(?:education|learning|stories|content)|"
    r"preschool(?:ers)?|pre-school|toddlers?|kindergarten|nursery\s+(?:rhymes|learning)|"
    r"learn\s+(?:the\s+)?(?:abc|alphabet|numbers|counting|colou?rs)\s+for\s+kids|"
    r"abc(?:s|['’]s)?\s+for\s+kids|nursery\s+rhymes)\b",
    re.IGNORECASE,
)


def classify_made_for_kids(title: str = "", description: str = "", script: Any = None) -> bool | None:
    """Return True only for strong child-directed evidence; None means uncertain.

    This deliberately does not infer child-directed status from the words 'facts',
    'story', 'animation', 'cartoon', or 'fantasy' alone.
    """
    parts = [title or "", description or ""]
    if isinstance(script, dict):
        for key in ("title", "topic", "description", "audience", "style", "format"):
            value = script.get(key)
            if isinstance(value, str):
                parts.append(value)
        scenes = script.get("scenes", [])
        if isinstance(scenes, list):
            for scene in scenes:
                if isinstance(scene, dict):
                    for key in ("narration", "text", "visual_prompt", "dialogue"):
                        value = scene.get(key)
                        if isinstance(value, str):
                            parts.append(value)
    elif isinstance(script, str):
        parts.append(script)
    text = " ".join(parts)
    return True if _CHILD_DIRECTED.search(text) else None
