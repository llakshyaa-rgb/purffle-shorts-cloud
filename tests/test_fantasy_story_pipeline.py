from purffle_shorts.script import MULTI_SPEAKER, normalize
from purffle_shorts.creative_director import apply_to_script


def test_fantasy_mode_uses_two_attributed_speakers_and_keeps_cast():
    data = {
        "topic": "The Moonlit Oath",
        "title": "The Moonlit Oath",
        "hook_text": "Never trust a dragon",
        "cast": ["Mira", "Kael"],
        "scenes": [
            {
                "speaker": "A",
                "narration": "Give me the key, Kael.",
                "search_query": "Mira demands key",
                "image_prompt": "REUSABLE CHARACTER: Mira — young mage, copper braid, teal cloak, brass star clasp\nSHOT: Mira holds out her hand.",
                "emphasis": ["key"],
                "emoji": "",
            },
            {
                "speaker": "B",
                "narration": "Only if you tell me why the king fears it.",
                "search_query": "Kael challenges Mira",
                "image_prompt": "REUSABLE CHARACTER: Kael — silver-haired elf, charcoal armor, broken silver ear cuff\nSHOT: Kael guards the key.",
                "emphasis": ["king"],
                "emoji": "",
            },
        ],
        "description": "An original fantasy scene.",
        "hashtags": ["#fantasy", "#shorts"],
        "tags": ["fantasy", "magic"],
        "category": "entertainment",
    }
    script = normalize(data, "The Moonlit Oath", "fantasy", "en")
    assert "fantasy" in MULTI_SPEAKER
    assert script.cast == ["Mira", "Kael"]
    assert [scene.speaker for scene in script.scenes] == ["A", "B"]
    assert script.multi_speaker


def test_director_preserves_canonical_character_bible_and_distinct_actions():
    data = {
        "topic": "The Moonlit Oath",
        "title": "The Moonlit Oath",
        "hook_text": "Never trust a dragon",
        "cast": ["Mira", "Kael"],
        "scenes": [
            {
                "speaker": "A",
                "narration": "Mira reaches for the key.",
                "search_query": "Mira reaches for key",
                "image_prompt": "REUSABLE CHARACTER: Mira — young mage, copper braid, teal cloak, brass star clasp\nSHOT: Mira reaches for the key.",
                "emphasis": [],
                "emoji": "",
            },
            {
                "speaker": "A",
                "narration": "Mira snatches the key as the guard lunges.",
                "search_query": "Mira snatches key guard lunges",
                "image_prompt": "REUSABLE CHARACTER: Mira — young mage, copper braid, teal cloak, brass star clasp\nSHOT: Mira snatches the key as the guard lunges.",
                "emphasis": [],
                "emoji": "",
            },
        ],
        "description": "An original fantasy scene.",
        "hashtags": ["#fantasy", "#shorts"],
        "tags": ["fantasy", "magic"],
        "category": "entertainment",
    }
    script = normalize(data, "The Moonlit Oath", "fantasy", "en")
    plan = [
        {"visual_prompt": "FANTASY CINEMATIC FOOTAGE: Mira reaches for the brass key.", "visual_mode": "footage"},
        {"visual_prompt": "FANTASY CINEMATIC FOOTAGE: Mira snatches the key as the guard lunges.", "visual_mode": "footage"},
    ]
    apply_to_script(script, plan)
    identity = "REUSABLE CHARACTER: Mira — young mage, copper braid, teal cloak, brass star clasp"
    assert identity in script.scenes[0].image_prompt
    assert identity in script.scenes[1].image_prompt
    assert "reaches for the brass key" in script.scenes[0].image_prompt
    assert "snatches the key as the guard lunges" in script.scenes[1].image_prompt
    assert script.scenes[0].image_prompt != script.scenes[1].image_prompt
