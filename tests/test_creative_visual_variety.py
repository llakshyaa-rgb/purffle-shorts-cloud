from __future__ import annotations

from purffle_shorts.creative_director import apply_to_script, direct, notes_markdown
from purffle_shorts.script import Scene, Script


def _script() -> Script:
    return Script(
        topic="Why the Moon stays in orbit",
        title="Why the Moon Doesn't Fall",
        hook_text="The Moon is falling",
        scenes=[
            Scene(
                narration="You might think the Moon is standing still.",
                search_query="Moon orbit around Earth",
                image_prompt="Moon and Earth",
            ),
            Scene(
                narration="Gravity pulls it inward while its sideways motion keeps it moving around Earth.",
                search_query="gravity and orbital motion",
                image_prompt="Moon orbit diagram",
            ),
        ],
        description="A simple explanation of lunar orbit.",
        style="explainer",
    )


def test_local_director_plans_every_scene_with_valid_choices():
    script = _script()
    plan = direct(script, None)
    assert len(plan) == len(script.scenes)
    assert [beat["scene"] for beat in plan] == [1, 2]
    assert all(beat["camera"] in {"push_in", "pull_out", "pan_left", "pan_right", "static", "shake"}
               for beat in plan)
    assert all(beat["visual_mode"] in {"illustration", "diagram", "footage", "photo", "symbolic"}
               for beat in plan)
    assert all(beat["reason"] for beat in plan)


def test_director_visual_prompts_are_applied_to_the_real_script():
    script = _script()
    plan = direct(script, None)
    apply_to_script(script, plan)
    assert all(scene.image_prompt for scene in script.scenes)
    assert any(tag in script.scenes[0].image_prompt for tag in (
        "DIAGRAM / EXPLANATORY GRAPHIC:", "ORIGINAL DRAWN ILLUSTRATION:",
        "SYMBOLIC VISUAL METAPHOR:", "REAL-WORLD DOCUMENTARY FOOTAGE:",
        "REALISTIC EDITORIAL PHOTOGRAPH:",
    ))


def test_director_notes_are_concise_and_match_scene_count():
    script = _script()
    plan = direct(script, None)
    notes = notes_markdown(script, plan)
    assert "Director's Notes" in notes
    assert "Scene 1" in notes and "Scene 2" in notes
    assert "not private chain-of-thought" in notes
    assert notes.count("**Why:**") == len(script.scenes)


def test_director_marks_reusable_archetypes_for_the_asset_library():
    script = _script()
    script.style = "story"
    script.scenes[0].narration = "The detective opened the hidden door."
    plan = direct(script, None)
    assert plan[0]["character"] == "purffle_detective"
    apply_to_script(script, plan)
    assert "REUSABLE CHARACTER: purffle_detective" in script.scenes[0].image_prompt
    assert "REUSABLE CHARACTER:" not in script.scenes[1].image_prompt


def test_factual_fallback_does_not_turn_generic_why_into_a_diagram():
    from purffle_shorts.script import Scene

    scene = Scene(
        narration="Why Saturn is less dense than water.",
        search_query="Saturn planet atmosphere",
        image_prompt="Saturn with its rings",
    )
    plan = direct(Script(topic="Saturn", title="Saturn", hook_text="", scenes=[scene],
                         description="", style="facts"), None)
    assert plan[0]["visual_mode"] == "footage"


def test_mechanism_can_still_use_a_diagram():
    from purffle_shorts.script import Scene

    scene = Scene(
        narration="Gravity pulls the Moon inward while its sideways motion keeps it in orbit.",
        search_query="gravity pulls moon orbit mechanism",
        image_prompt="Moon orbit around Earth",
    )
    plan = direct(Script(topic="Orbit", title="Orbit", hook_text="", scenes=[scene],
                         description="", style="explainer"), None)
    assert plan[0]["visual_mode"] == "diagram"
