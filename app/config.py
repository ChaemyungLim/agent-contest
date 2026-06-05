from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model paths — 사내 모델 레지스트리에서 받은 경로로 .env에서 덮어쓰기
    embedder_path: str = "models/bge-m3"
    reranker_path: str = "models/bge-reranker-v2-m3"

    # ChromaDB (임베디드, 영속화)
    chroma_persist_dir: str = "data/chroma"
    collection_official: str = "official"
    collection_history: str = "history"

    # Indexing
    history_months: int = 12  # 시점 컷
    dedup_threshold: float = 0.95
    embed_batch_size: int = 32

    # Retrieval
    top_k_official: int = 5
    top_k_history: int = 15
    rerank_top_k: int = 20
    final_top_k: int = 3
    rrf_k: int = 60
    official_boost: float = 0.10  # legacy search() 전용 (tiered 모드에선 미사용)
    answer_threshold: float = 0.5

    # LLM 종합
    synthesis_top_k: int = 3  # tiered 후 LLM에 넣을 최대 사례 수

    # Routing
    departments_yaml: str = "app/routing/departments.yaml"

    # Excel column mapping
    col_question: str = "question"
    col_answer: str = "answer"
    col_answered_at: str = "answered_at"
    col_answered_by: str = "answered_by"


settings = Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    """프로젝트 루트 기준 상대 경로를 절대 경로로 정규화."""
    path = Path(p)
    return path if path.is_absolute() else project_root() / path
