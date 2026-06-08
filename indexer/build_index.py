"""인덱싱 CLI (ChromaDB 임베디드).

폐쇄망 + CPU 환경에서 수 시간 소요될 수 있다.
배포 자동화 단계가 아닌 별도 잡으로 실행하라.

사용:
  python -m indexer.build_index \
      --official data/official_faq.xlsx \
      --history data/history_qa.xlsx \
      --months 12

옵션:
  --skip-history   : official만 인덱싱 (MVP)
  --rebuild        : 기존 collection 삭제 후 재생성
  --no-dedup       : history dedup 생략 (디버그)
"""
from __future__ import annotations

import argparse
import sys

import chromadb
from chromadb.errors import NotFoundError
from tqdm import tqdm

from app.config import settings
from app.retrieval.embedder import Embedder
from app.retrieval.searcher import bm25_index_path, build_bm25_index, get_chroma_client
from app.schemas import Hit
from indexer.dedup import dedup_records
from indexer.load_excel import load_excel


def _ensure_collection(
    client: chromadb.api.ClientAPI, name: str, rebuild: bool
) -> chromadb.Collection:
    if rebuild:
        try:
            client.delete_collection(name)
        except (NotFoundError, ValueError):
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _stream_upsert(
    collection: chromadb.Collection,
    records: list[Hit],
    embedder: Embedder,
    batch_size: int = 256,
) -> None:
    """배치 단위로 인코딩 + 업서트 → 30만 행 전체 임베딩을 메모리에 동시 보유 방지."""
    print(f"  encode+upsert {len(records)} → '{collection.name}'...")
    for start in tqdm(range(0, len(records), batch_size), desc=f"  {collection.name}"):
        batch = records[start : start + batch_size]
        vecs = embedder.encode_passages(
            [r.question for r in batch], batch_size=batch_size, show_progress=False
        )
        collection.upsert(
            ids=[r.id for r in batch],
            embeddings=vecs.tolist(),
            metadatas=[
                {
                    "doc_id": r.id,
                    "question": r.question,
                    "answer": r.answer,
                    "source": r.source,
                    "answered_at": r.answered_at.isoformat() if r.answered_at else "",
                }
                for r in batch
            ],
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FAQ RAG 인덱싱 (ChromaDB)")
    ap.add_argument("--official", required=True, help="공식 FAQ 엑셀 경로")
    ap.add_argument("--history", help="과거 Q&A 엑셀 경로 (생략 가능)")
    ap.add_argument("--months", type=int, default=settings.history_months,
                    help=f"history 시점 컷 개월 (기본 {settings.history_months})")
    ap.add_argument("--skip-history", action="store_true", help="MVP: history 인덱싱 생략")
    ap.add_argument("--rebuild", action="store_true", help="기존 collection 삭제 후 재생성")
    ap.add_argument("--no-dedup", action="store_true", help="history dedup 생략")
    args = ap.parse_args(argv)

    embedder = Embedder()
    client = get_chroma_client()

    print(f"== load official: {args.official}")
    official_records = list(load_excel(args.official, source="official"))
    if not official_records:
        print("ERROR: 공식 FAQ 레코드가 0건입니다.", file=sys.stderr)
        return 1

    history_records: list[Hit] = []
    if not args.skip_history:
        if not args.history:
            print("WARN: --history 미지정. official만 인덱싱합니다.")
        else:
            print(f"== load history: {args.history} (cutoff={args.months}개월)")
            history_records = list(
                load_excel(args.history, source="history", months_cutoff=args.months)
            )
            if history_records and not args.no_dedup:
                history_records = [
                    r for r in dedup_records(history_records, embedder) if r.source == "history"
                ]

    print(f"== Chroma collections (rebuild={args.rebuild})")
    official_col = _ensure_collection(client, settings.collection_official, args.rebuild)
    _stream_upsert(official_col, official_records, embedder)

    if history_records:
        history_col = _ensure_collection(client, settings.collection_history, args.rebuild)
        _stream_upsert(history_col, history_records, embedder)
    elif args.rebuild:
        try:
            client.delete_collection(settings.collection_history)
        except (NotFoundError, ValueError):
            pass

    print("== build BM25 index")
    all_records = official_records + history_records
    bm25 = build_bm25_index(all_records)
    bm25_path = bm25_index_path()
    bm25.save(bm25_path)
    print(f"  saved BM25 index: {bm25_path} ({len(all_records)} docs)")

    print(f"\n✅ Done. official={len(official_records)} history={len(history_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
