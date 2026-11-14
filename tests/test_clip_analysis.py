"""AI clip analysis shortlist tests: ranking and bounded candidate counts."""

from purffle_shorts.clipper import Sentence, pick_clips
from purffle_shorts.config import Settings


class FakeClipLLM:
    label = "test:fake"

    def complete_json(self, system, user, schema, max_tokens, temperature):
        clips = []
        for i in range(10):
            start = float(i * 7)
            clips.append({
                "start": start, "end": start + 6,
                "title": f"Moment {i + 1}", "hook_text": f"Hook {i + 1}",
                "description": "A complete moment. What do you think?",
                "hashtags": ["#test"], "tags": ["test"], "score": 100 - i,
                "reason": f"Reason {i + 1}",
            })
        return {"clips": clips}


def test_clip_analysis_can_return_ten_ranked_moments():
    sents = [Sentence(i * 7.0, i * 7.0 + 6.0, f"Sentence {i + 1}.") for i in range(10)]
    plans = pick_clips(FakeClipLLM(), sents, Settings(), 10, min_s=5, max_s=8)
    assert len(plans) == 10
    assert [p.score for p in plans] == list(range(100, 90, -1))
    assert plans[0].title == "Moment 1"
    assert plans[-1].title == "Moment 10"


def test_clip_analysis_respects_requested_shortlist_size():
    sents = [Sentence(i * 7.0, i * 7.0 + 6.0, f"Sentence {i + 1}.") for i in range(10)]
    plans = pick_clips(FakeClipLLM(), sents, Settings(), 4, min_s=5, max_s=8)
    assert len(plans) == 4


def test_manga_edit_grade_is_high_contrast_monochrome():
    from purffle_shorts.render import GRADES

    assert "hue=s=0" in GRADES["manga"]
    assert "unsharp" in GRADES["manga"]


def test_native_visual_ranker_prefers_active_frames_without_an_llm():
    from purffle_shorts.clipper import rank_native_visual_moments

    features = []
    for second in range(0, 121, 2):
        active = 20 <= second <= 62
        features.append({
            "time": float(second),
            "change": 0.75 if active and second % 6 == 0 else (0.38 if active else 0.02),
            "edge": 0.8 if active else 0.15,
            "contrast": 0.75 if active else 0.2,
            "colour": 0.8 if active else 0.25,
        })
    plans = rank_native_visual_moments(features, 120, 5, 20, 40)
    assert plans
    assert plans[0].score is not None
    assert "Native pixel analysis" in plans[0].reason
    assert all(plan.end - plan.start >= 20 for plan in plans)
    assert all(not (a.start < b.end and a.end > b.start)
               for i, a in enumerate(plans) for b in plans[i + 1:])


def test_visual_context_scoring_adds_face_eye_and_animation_signals():
    from purffle_shorts.clipper import apply_visual_context_scores

    frames = [{"time": 0.0, "change": 0.1}, {"time": 2.0, "change": 0.5}]
    apply_visual_context_scores(frames, {"face_frames": 1, "eye_frames": 1, "animation_like_frames": 1})
    assert frames[0]["face_signal"] == 0.5
    assert frames[0]["eye_signal"] == 0.5
    assert frames[0]["animation_texture"] == 0.5
