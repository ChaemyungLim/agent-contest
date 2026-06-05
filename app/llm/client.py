"""사내 LLM 호출 인터페이스.

이 모듈은 사내망 LLM 엔드포인트와 연결하는 단일 진입점이다.
실제 호출 구현은 사내 인프라에 맞게 채워 넣어야 한다.

MVP에서는 LLM 사용이 선택적이다 (답변 원문을 그대로 노출하는 게 기본).
- 여러 후보 답변을 종합 설명하고 싶을 때
- 추후 query rewriting / HyDE 등을 시도할 때
이 함수만 구현하면 동작한다.
"""
from typing import Any


def call_llm(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
    **kwargs: Any,
) -> str:
    """사내 LLM 호출.

    Args:
        messages: OpenAI 호환 메시지 리스트.
            예: [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
        temperature: 샘플링 온도
        max_tokens: 응답 최대 토큰

    Returns:
        LLM 응답 텍스트
    """
    raise NotImplementedError(
        "사내 LLM 엔드포인트 연결이 필요합니다. "
        "app/llm/client.py 의 call_llm() 함수를 구현하세요."
    )
