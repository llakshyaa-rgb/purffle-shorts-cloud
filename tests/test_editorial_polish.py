from purffle_shorts.pipeline import trimmed_voice_duration


def test_trimmed_voice_duration_removes_long_trailing_silence():
    assert trimmed_voice_duration(26.0, [(4.0, 4.4), (20.0, 26.0)]) == 20.1


def test_trimmed_voice_duration_keeps_short_natural_breath():
    assert trimmed_voice_duration(20.0, [(19.7, 20.0)]) == 20.0


def test_trimmed_voice_duration_ignores_internal_silence():
    assert trimmed_voice_duration(20.0, [(8.0, 9.0)]) == 20.0


def test_trimmed_voice_duration_handles_missing_audio_pauses():
    assert trimmed_voice_duration(20.0, []) == 20.0
