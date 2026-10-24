

def test_stock_without_descriptive_metadata_is_not_approved_as_relevant():
    item = M.MediaItem("pexels", "opaque-id", "video", width=1080, height=1920)
    # Neutral relevance is useful for ranking, but must not make an undocumented asset pass
    # the stricter candidate-admission gate in Visuals.for_scene.
    assert M.relevance(item, "Saturn planet rings") == 0.5
    assert not (
        (item.text or item.extra.get("description") or item.extra.get("categories"))
        and M.relevance(item, "Saturn planet rings") >= 0.45
    )


def test_stock_with_unrelated_metadata_does_not_match_scene():
    item = M.MediaItem("pexels", "opaque-id", "video", text="city traffic at night")
    assert M.relevance(item, "Saturn planet rings") < 0.45
