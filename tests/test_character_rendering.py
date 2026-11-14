from purffle_shorts.media import character_rendering_prompt


def test_character_prompt_enforces_human_anatomy_and_eye_alignment():
    prompt = character_rendering_prompt("REUSABLE CHARACTER: Mira — original adult character")
    assert "aligned pupils" in prompt
    assert "five correct" in prompt
    assert "no merged bodies" in prompt


def test_character_prompt_uses_reference_as_identity_source():
    prompt = character_rendering_prompt("REUSABLE CHARACTER: Mira", has_reference=True)
    assert "model sheet as the identity source" in prompt
    assert "Do not redesign, recast" in prompt


def test_character_prompt_does_not_require_reference_for_textual_bible():
    prompt = character_rendering_prompt("REUSABLE CHARACTER: Mira", has_reference=False)
    assert "identity description stable" in prompt
