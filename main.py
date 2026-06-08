"""FastAPI 앱 진입점.

배포 환경에서 `uvicorn main:app`으로 기동.
"""
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.retrieval.embedder import get_embedder
from app.retrieval.reranker import get_reranker
from app.retrieval.searcher import get_searcher
from app.routing.department import get_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedder = get_embedder()
    reranker = get_reranker()
    get_searcher()
    get_router()
    # PyTorch lazy init/JIT 워밍업: 첫 실제 요청이 cold-start 페널티 안 받도록
    embedder.encode_query("워밍업")
    reranker.score("워밍업", ["워밍업 문장"])
    yield


app = FastAPI(
    title="장기인수 FAQ RAG 챗봇",
    description="지점 매니저용 사전 문의 자동응답 (공식 FAQ + 과거 Q&A 검색)",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(chat_router)
