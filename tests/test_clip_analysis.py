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
