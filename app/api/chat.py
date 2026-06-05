"""POST /chat — FAQ RAG 챗봇 메인 엔드포인트.

흐름:
1. tiered 검색: 공식FAQ 우선, 부족하면 과거Q&A로 채움 (임계치 기반)
2. 통과 사례 0건 → 부서 라우팅
3. 사례 있음 → LLM 종합
   3-1. LLM이 답변 불가 판단(INSUFFICIENT_CONTEXT) → 부서 라우팅
   3-2. 완전/부분 답변 → matched 응답
"""
from fastapi import APIRouter

from app.config import settings
from app.llm.synthesizer import synthesize
from app.retrieval.searcher import Searcher, get_searcher
from app.routing.department import get_router
from app.schemas import ChatRequest, ChatResponse, Hit, SourceInfo


router = APIRouter()


def _tiered_search(searcher: Searcher, query: str) -> list[Hit]:
    """공식 우선 → 부족하면 과거로 채움. 모두 임계치 통과한 것만."""
    needed = settings.synthesis_top_k
    threshold = settings.answer_threshold

    official = searcher.search_by_source(query, source="official")
    official_passed = [h for h in official if h.score >= threshold]
    if len(official_passed) >= needed:
        return official_passed[:needed]

    history = searcher.search_by_source(query, source="history")
    history_passed = [h for h in history if h.score >= threshold]
    return (official_passed + history_passed)[:needed]


def _route_to_dept(query: str, alternatives: list[Hit]) -> ChatResponse:
    dept = get_router().route(query, top_hits=alternatives or None)
    return ChatResponse(
        matched=False,
        department=dept.to_info(),
        message=f"유사 사례를 찾지 못했습니다. 담당 부서는 '{dept.name}' 입니다.",
        alternatives=alternatives,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    searcher = get_searcher()
    candidates = _tiered_search(searcher, req.query)

    if not candidates:
        return _route_to_dept(req.query, alternatives=[])

    result = synthesize(req.query, candidates)
    if result.answer is None:
        # LLM이 답변 불가로 판단 → 검색 사례는 alternatives로 보존
        return _route_to_dept(req.query, alternatives=candidates)

    top = candidates[0]
    return ChatResponse(
        matched=True,
        answer=result.answer,
        answer_type=result.answer_type,
        source=SourceInfo(
            type=top.source,
            answered_at=top.answered_at,
            answered_by=top.answered_by,
        ),
        confidence=top.score,
        alternatives=candidates,
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
