from purffle_shorts.creative_director import _animation_style, _character_bible, apply_to_script, direct
from purffle_shorts.script import Scene, Script


def _anime_script():
    return Script(
        topic="Anime-style fantasy duel",
        title="The Last Ember",
        hook_text="A duel begins",
        scenes=[
            Scene(narration="Aiko raises her blade.", search_query="anime swordswoman", image_prompt="Aiko raises a blade"),
            Scene(narration="Aiko blocks the strike.", search_query="anime sword clash", image_prompt="Aiko blocks a strike"),
        ],
        description="An original animated short.",
        style="story",
        cast=["Aiko"],
    )


def test_anime_style_is_detected_without_vendor_api():
    key, prompt = _animation_style(_anime_script())
    assert key == "anime"
    assert "cel shading" in prompt


def test_character_bible_is_stable_for_same_episode_character():
    script = _anime_script()
    assert _character_bible(script, "Aiko", "anime") == _character_bible(script, "Aiko", "anime")


def test_anime_style_and_character_identity_reach_scene_prompts():
    script = _anime_script()
    apply_to_script(script, direct(script, None))
    assert all("ANIMATION STYLE LOCK: anime" in scene.image_prompt for scene in script.scenes)
    assert all("REUSABLE CHARACTER: Aiko" in scene.image_prompt for scene in script.scenes)
