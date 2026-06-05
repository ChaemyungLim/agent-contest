"""Synthesizer 파싱 로직 테스트. call_llm은 monkeypatch로 모킹."""
from __future__ import annotations

import pytest

from app.llm import synthesizer
from app.llm.synthesizer import (
    INSUFFICIENT_MARKER,
    PARTIAL_MARKER,
    SynthesisResult,
    synthesize,
)


def _patch_llm(monkeypatch: pytest.MonkeyPatch, response: str) -> list[dict]:
    """call_llm을 고정 응답으로 모킹. 인자 캡쳐용 리스트 반환."""
    captured: list[dict] = []

    def fake(messages, **kwargs):
        captured.append({"messages": messages, "kwargs": kwargs})
        return response

    monkeypatch.setattr(synthesizer, "call_llm", fake)
    return captured


def test_empty_hits_returns_no_answer(make_hit):
    result = synthesize("질문", hits=[])
    assert result == SynthesisResult(answer=None, answer_type=None)


def test_full_answer(monkeypatch, make_hit):
    _patch_llm(monkeypatch, "환급율 조회는 그룹웨어에서 가능합니다. [참고: #1]")
    hits = [make_hit(id="a", source="official", question="환급율 조회 방법")]

    result = synthesize("환급율 어떻게 봐요", hits=hits)

    assert result.answer is not None
    assert result.answer_type == "full"
    assert "[참고: #1]" in result.answer


def test_partial_answer_detected(monkeypatch, make_hit):
    response = (
        "환급율 조회는 그룹웨어 환급율조회 메뉴에서 가능합니다. [참고: #1]\n"
        f"\n{PARTIAL_MARKER}\n"
        "- 환급율 산정 공식\n"
        "→ 장기상품개발파트에 추가 문의 권장"
    )
    _patch_llm(monkeypatch, response)
    hits = [make_hit(id="a", source="official")]

    result = synthesize("환급율 조회와 산정 공식", hits=hits)

    assert result.answer is not None
    assert result.answer_type == "partial"
    assert PARTIAL_MARKER in result.answer


def test_insufficient_context_escalates(monkeypatch, make_hit):
    _patch_llm(monkeypatch, INSUFFICIENT_MARKER)
    hits = [make_hit(id="a")]

    result = synthesize("뜬금없는 질문", hits=hits)

    assert result == SynthesisResult(answer=None, answer_type=None)


def test_insufficient_context_with_trailing_text(monkeypatch, make_hit):
    """LLM이 'INSUFFICIENT_CONTEXT 입니다' 같이 변형 출력해도 escalate."""
    _patch_llm(monkeypatch, f"{INSUFFICIENT_MARKER} (사례가 부족함)")
    hits = [make_hit(id="a")]

    result = synthesize("질문", hits=hits)

    assert result.answer is None


def test_prompt_includes_hits(monkeypatch, make_hit):
    captured = _patch_llm(monkeypatch, "ok. [참고: #1]")
    hits = [
        make_hit(id="a", source="official", question="공식 질문 A", answer="공식 답변 A"),
        make_hit(id="b", source="history", question="과거 질문 B", answer="과거 답변 B"),
    ]

    synthesize("쿼리", hits=hits)

    assert len(captured) == 1
    user_msg = captured[0]["messages"][1]["content"]
    assert "공식 질문 A" in user_msg
    assert "공식 답변 A" in user_msg
    assert "과거 질문 B" in user_msg
    assert "공식FAQ" in user_msg
    assert "과거Q&A" in user_msg
