from app.retrieval.text import tokenize


def test_korean_tokenization():
    tokens = tokenize("신계약 고지의무 위반 case-12")
    assert "신계약" in tokens
    assert "고지의무" in tokens
    assert "위반" in tokens
    assert "case" in tokens
    assert "12" in tokens


def test_empty():
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


def test_lowercase():
    assert tokenize("APPLE Banana") == ["apple", "banana"]


def test_punct_strip():
    assert "별표3" in tokenize("별표3, 심사기준표 참조")
