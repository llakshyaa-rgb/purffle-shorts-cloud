from purffle_shorts.audience import classify_made_for_kids


def test_child_directed_learning_is_made_for_kids():
    assert classify_made_for_kids("ABC for kids: learn numbers") is True


def test_general_facts_are_not_automatically_child_directed():
    assert classify_made_for_kids("Amazing facts about the ocean") is None


def test_general_fantasy_is_not_automatically_child_directed():
    assert classify_made_for_kids("The dragon who saved a kingdom", script={"scenes": [
        {"narration": "A dragon wakes beneath the mountain."}
    ]}) is None


def test_explicit_child_audience_in_script_is_detected():
    assert classify_made_for_kids("Color facts", script={"audience": "educational stories for young children"}) is True
