from purffle_shorts.creative_director import direct
from purffle_shorts.script import Scene, Script


def test_factual_fallback_does_not_turn_generic_why_into_a_diagram():
    scene = Scene(
        narration="Why Saturn is less dense than water.",
        search_query="Saturn planet atmosphere",
        image_prompt="Saturn with its rings",
    )
    script = Script(topic="Saturn", title="Saturn", hook_text="", scenes=[scene],
                    description="", style="facts")
    assert direct(script, None)[0]["visual_mode"] == "footage"


def test_mechanism_can_still_use_a_diagram():
    scene = Scene(
        narration="Gravity pulls the Moon inward while its sideways motion keeps it in orbit.",
        search_query="gravity pulls moon orbit mechanism",
        image_prompt="Moon orbit around Earth",
    )
    script = Script(topic="Orbit", title="Orbit", hook_text="", scenes=[scene],
                    description="", style="explainer")
    assert direct(script, None)[0]["visual_mode"] == "diagram"
