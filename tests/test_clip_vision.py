"""Multimodal clip selection tests using timestamped visual storyboard samples."""

from purffle_shorts.clipper import Sentence, pick_visual_clips
from purffle_shorts.config import Settings


class FakeVisionLLM:
    def complete_vision_json(self, system, user, images, schema, max_tokens, temperature):
        assert len(images) == 3
        assert "frame" in user.lower()
        return {"clips": [
            {"start": 10, "end": 35, "title": "Visual payoff", "hook_text": "Watch this",
             "description": "A clear action. What happens next?", "hashtags": ["#anime"], "tags": ["anime"],
             "score": 92, "reason": "The action builds to a clear payoff."},
            {"start": 40, "end": 65, "title": "Second scene", "hook_text": "Unexpected turn",
             "description": "A second moment. What do you think?", "hashtags": [], "tags": [],
             "score": 85, "reason": "Distinct scene and payoff."},
        ]}


def test_visual_analysis_selects_distinct_moments_without_transcript():
    images = [{"timestamp": float(i * 20), "mime": "image/jpeg", "data": "AA=="} for i in range(3)]
    plans = pick_visual_clips(FakeVisionLLM(), images, 100, [], Settings(), 10, 20, 58)
    assert [p.title for p in plans] == ["Visual payoff", "Second scene"]
    assert [p.score for p in plans] == [92, 85]


def test_visual_analysis_aligns_candidates_to_transcript_sentences():
    images = [{"timestamp": float(i * 20), "mime": "image/jpeg", "data": "AA=="} for i in range(3)]
    sents = [Sentence(i * 5.0, i * 5.0 + 4.0, f"Line {i}.") for i in range(20)]
    plans = pick_visual_clips(FakeVisionLLM(), images, 100, sents, Settings(), 10, 20, 58)
    assert plans
    assert all(any(abs(p.start - s.start) < 0.01 for s in sents) for p in plans)
    assert all(any(abs(p.end - s.end) < 0.01 for s in sents) for p in plans)
