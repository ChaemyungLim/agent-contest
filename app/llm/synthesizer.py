"""LLM 답변 종합.

검색 단계 통과한 사례 N개를 LLM에 넣어 일관된 답변 생성.

LLM 출력 형태 3가지:
1. 완전 답변          : 사례로 질문 전체 답변 가능
2. 부분 답변          : 일부만 답변 + 미답변 항목은 부서 안내 힌트
3. INSUFFICIENT_CONTEXT: 답변 불가 (호출자가 부서 라우팅으로 폴백)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.llm.client import call_llm
from app.schemas import Hit


INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"
PARTIAL_MARKER = "[부분답변]"


SYSTEM_PROMPT = f"""당신은 KB손해보험 장기인수팀 직원입니다.
지점 매니저의 업무 문의에 대해, 아래 제공된 과거 답변 사례만을 근거로 답변하세요.

[준수 규칙]
1. 사례에 명시된 정보 외에 추가하지 마세요. (환각 금지)
2. 여러 사례가 동일한 내용을 다루면 한 번에 정리해 답변하세요.
3. 사례 간 내용이 다르면, 최신 답변(answered_at 최신순)을 우선하되 상충 사실을 간단히 언급하세요.
4. 매니저-인수직원 간 자연스러운 업무 톤. 인사말/맺음말 없이 본론만.
5. 답변 본문 끝에 사용한 사례 번호를 [참고: #1, #3] 형식으로 표시.

[부분 답변]
질문의 일부만 사례로 답변 가능할 때:
- 답변 가능한 부분만 작성 (위 규칙 1~5 동일 적용)
- 그 다음 빈 줄 + 마커 그대로 출력: {PARTIAL_MARKER}
- 마커 아래에 미답변 항목을 한 줄에 하나씩 짧게 나열
- 부분 답변 전체 합쳐 6문장 이하
- 사례 인용 [참고: #N]이 없으면 부분 답변 자격 없음 → {INSUFFICIENT_MARKER} 출력

[답변 완전 불가]
다음 경우엔 정확히 아래 문자열만 출력 (추가 설명 금지):
{INSUFFICIENT_MARKER}
- 사례들이 질문의 핵심을 다루지 않을 때
- 사례 간 심각한 모순으로 일관된 답변이 불가능할 때
- 사례 정보로는 답변 불가능한 영역의 질문일 때
"""


@dataclass
class SynthesisResult:
    """LLM 종합 결과.

    - answer=None : 답변 불가, 호출자는 부서 라우팅으로 폴백
    - answer_type : matched 시 "full" 또는 "partial"
    """
    answer: str | None
    answer_type: Literal["full", "partial"] | None


def _format_cases(hits: Sequence[Hit]) -> str:
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        if h.source == "official":
            tag = "공식FAQ"
        else:
            tag = f"과거Q&A · {h.answered_at}" if h.answered_at else "과거Q&A"
        lines.append(f"#{i} ({tag})\n질문: {h.question}\n답변: {h.answer}")
    return "\n\n".join(lines)


def _build_user_message(query: str, hits: Sequence[Hit]) -> str:
    return f"""문의: {query}

참고 사례:
{_format_cases(hits)}

위 사례를 참고해 답변하세요."""


def synthesize(query: str, hits: Sequence[Hit]) -> SynthesisResult:
    """LLM 호출 후 응답을 파싱해 SynthesisResult로 반환.

    호출자는 answer is None을 부서 라우팅 폴백 신호로 처리.
    """
    if not hits:
        return SynthesisResult(answer=None, answer_type=None)

    response = call_llm(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, hits)},
        ],
        temperature=0.2,
    ).strip()

    # 응답 변형 허용: "INSUFFICIENT_CONTEXT" 또는 "INSUFFICIENT_CONTEXT 입니다" 등
    if response.upper().startswith(INSUFFICIENT_MARKER):
        return SynthesisResult(answer=None, answer_type=None)

    answer_type: Literal["full", "partial"] = (
        "partial" if PARTIAL_MARKER in response else "full"
    )
    return SynthesisResult(answer=response, answer_type=answer_type)
