import json
from pathlib import Path

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.llm import extract_json
from purffle_shorts.script import SCRIPT_SCHEMA, build_prompt, load_script, normalize, offline_script


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here it is:\n{"a": 1}\nHope that helps.',
    '{"a": 1,}',
])
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_normalize_cleans_everything():
    data = {
        "topic": "Octopus", "title": '"Octopus Secrets" #shorts 🐙', "hook_text": "3 HEARTS?!",
        "scenes": [
            {"narration": "[Hook] Narrator: Octopuses have three hearts! 🐙 #facts", "search_query": "octopus",
             "image_prompt": "an octopus"},
            {"narration": "", "search_query": "x", "image_prompt": "x"},
            "Plain string scene works too.",
        ],
        "description": "Wow. #octopus", "hashtags": ["octopus", "#Ocean Life", "#shorts", "#octopus"],
        "tags": ["a" * 70] + [f"tag{i}" for i in range(200)], "category": "nonsense",
    }
    s = normalize(data, "octopus", "facts", "en")
    assert s.scenes[0].narration == "Octopuses have three hearts!"
    assert len(s.scenes) == 2
    assert s.title == "Octopus Secrets"
    assert s.hashtags[0] == "#shorts" and "#OceanLife" in s.hashtags
    assert len([h for h in s.hashtags if h.lower() == "#octopus"]) == 1
    assert sum(len(t) + 1 for t in s.tags) <= 500
    assert s.category == "education" and s.category_id == "27"
    assert "#" not in s.description


def test_normalize_merges_too_many_scenes():
    data = {"scenes": [{"narration": f"Sentence {i}.", "search_query": "q", "image_prompt": "p"} for i in range(20)]}
    s = normalize(data, "x", "facts", "en")
    # Non-chat scripts cap at 16 scenes; preserve all 20 spoken sentences by merging
    # the shortest same-speaker neighbours rather than dropping narration.
    assert len(s.scenes) == 16
    assert s.narration.count("Sentence") == 20


def test_offline_script_is_complete():
    s = offline_script("", Settings())
    assert s.title and len(s.scenes) >= 4 and s.hashtags[0] == "#shorts"


def test_prompt_mentions_language_and_length():
    system, user = build_prompt("space", "facts", Settings(language="es", target_seconds=30), ["Old title"])
    assert "Spanish" in user and "word" in user.lower() and "Old title" in user
    assert "JSON" in system
    assert set(SCRIPT_SCHEMA["required"]) >= {"title", "scenes", "hashtags"}


def test_load_script_cleans_like_an_ai_script(tmp_path):
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({"title": "My Own Script #viral", "style": "not-a-style", "hashtags": ["cats"],
                             "scenes": [{"narration": "Cats sleep a lot.", "search_query": "sleeping cat"}]}))
    sc = load_script(f, Settings())
    assert sc.title == "My Own Script" and sc.style == "facts"
    assert sc.hashtags[:2] == ["#shorts", "#cats"] and sc.topic == "My Own Script #viral"
    assert sc.scenes[0].image_prompt == "sleeping cat"


def test_load_script_rejects_non_objects(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_script(f, Settings())


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "examples").glob("*.json")))
def test_shipped_examples_load(path):
    sc = load_script(path, Settings())
    assert len(sc.scenes) >= 4 and sc.hook_text and all(s.image_prompt for s in sc.scenes)


def test_narration_cleanup_turns_dashes_into_clean_pauses():
    from purffle_shorts.utils import clean_narration
    assert clean_narration("packed with copper — not iron.") == "packed with copper, not iron."
    assert clean_narration("blue–green , really ?") == "blue, green, really?"
    assert clean_narration("It ends here —.") == "It ends here."
    assert clean_narration("[Hook] Wow! #facts 🐙") == "Wow!"


def test_prompt_includes_creator_creative_direction():
    _, user = build_prompt(
        "Lake Natron", "story", Settings(), [],
        creative_brief="Use a cinematic cold open, build a mystery, and prioritize real close-up footage.",
    )
    assert "CREATOR CREATIVE DIRECTION" in user
    assert "cinematic cold open" in user
    assert "prioritize real close-up footage" in user



def test_lint_splits_sentences_correctly():
    from purffle_shorts.script import Scene, Script, lint_script

    script = Script(
        topic="test", title="A test", hook_text="A strange door.",
        scenes=[
            Scene("A strange door appeared. I opened it.", "tiny door in wardrobe",
                  "A tiny wooden door behind a wardrobe, close-up shot."),
            Scene("Inside was a note. It had my name.", "handwritten note on tiny cot",
                  "A handwritten note resting on a tiny cot, overhead shot."),
        ],
        description="", style="story", language="en",
    )
    checks = lint_script(script, Settings(target_seconds=10))
    assert not any("word sentence" in item for item in checks)


def test_lint_flags_repeated_visual_search_and_weak_prompt():
    from purffle_shorts.script import Scene, Script, lint_script

    script = Script(
        topic="test", title="A test", hook_text="A strange door.",
        scenes=[
            Scene("A strange door appeared.", "tiny door", "tiny door"),
            Scene("I opened it carefully.", "tiny door", "A door"),
        ],
        description="", style="story", language="en",
    )
    checks = lint_script(script, Settings(target_seconds=10))
    assert any("same footage search" in item for item in checks)
    assert any("underspecified visual prompt" in item for item in checks)


def test_script_prompt_sets_strong_runtime_length_target():
    settings = Settings.from_env()
    settings.target_seconds = 25
    system, user = build_prompt("Saturn density", "facts", settings, [])
    assert "HARD LENGTH TARGET" in user
    assert "25-second video" in user
