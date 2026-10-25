from purffle_shorts.config import Settings
from purffle_shorts.creative_director import apply_to_script
from purffle_shorts.script import build_prompt, normalize


def test_fantasy_keeps_story_specific_ensemble_and_speaker_mapping():
    data = {
        "topic": "The moon thief",
        "title": "The Moon Thief",
        "hook_text": "Someone stole the moon.",
        "cast": ["Neris", "Vey", "Captain Orro", "Sable"],
        "scenes": [
            {"speaker": "A", "narration": "Neris, give me the map.", "search_query": "Neris map",
             "image_prompt": "Neris asks Vey for a map."},
            {"speaker": "B", "narration": "Only if you tell me the truth.", "search_query": "Vey map",
             "image_prompt": "Vey holds a folded map."},
            {"speaker": "C", "narration": "Both of you, the guards are here!", "search_query": "Captain Orro guards",
             "image_prompt": "Captain Orro warns the pair."},
            {"speaker": "D", "narration": "Then we leave by the flooded tunnel.", "search_query": "Sable tunnel",
             "image_prompt": "Sable points toward a flooded tunnel."},
        ],
    }
    script = normalize(data, "The moon thief", "fantasy", "en")

    assert script.cast == ["Neris", "Vey", "Captain Orro", "Sable"]
    assert [scene.speaker for scene in script.scenes] == ["A", "B", "C", "D"]
    assert [script.speaker_name(scene.speaker) for scene in script.scenes] == [
        "Neris", "Vey", "Captain Orro", "Sable"
    ]

    plan = [
        {"visual_mode": "footage", "visual_prompt": scene.image_prompt}
        for scene in script.scenes
    ]
    apply_to_script(script, plan)
    for scene, name in zip(script.scenes, script.cast):
        assert f"REUSABLE CHARACTER: {name}" in scene.image_prompt


def test_fantasy_prompt_requires_fresh_purposeful_cast_not_recycled_roles():
    system, user = build_prompt("A witch bargaining with a living storm", "fantasy", Settings(), [])
    assert "Invent a fresh cast specifically for this exact prompt" in system
    assert "never reuse a default cast" in system
    assert "character must affect events" in system.lower()
