from app.retrieval.threshold import decide


def test_empty_falls_back():
    d = decide([])
    assert d.fallback is True
    assert d.answer is None


def test_below_threshold_falls_back(make_hit):
    d = decide([make_hit(id="a", score=0.3), make_hit(id="b", score=0.2)])
    assert d.fallback is True
    assert d.answer is None
    assert len(d.alternatives) == 2


def test_above_threshold_returns_top(make_hit):
    d = decide([
        make_hit(id="a", score=0.8),
        make_hit(id="b", score=0.6),
        make_hit(id="c", score=0.4),
    ])
    assert d.fallback is False
    assert d.answer is not None
    assert d.answer.score == 0.8
    assert len(d.alternatives) == 2
