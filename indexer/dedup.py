"""유사 질문 dedup: 같은 질문의 다른 시점 답변이 다수 → 최신 1개만 유지.

알고리즘:
1. 질문 임베딩 (배치 인코딩)
2. greedy 클러스터링: 정렬된 순서대로 각 질문을 기존 클러스터 representative와 비교 → 유사도 ≥ threshold면 같은 클러스터
3. 클러스터 내에서 answered_at 최신 1개만 채택

성능 노트: rep_vecs를 preallocated numpy 버퍼로 유지 → 30만 행 규모에서
매 iteration np.stack(N개) 재할당 폭주 방지. 더 큰 규모 (>1M)는 FAISS IVF 권장.
"""
from __future__ import annotations

from datetime import date

import numpy as np
from tqdm import tqdm

from app.config import settings
from app.retrieval.embedder import Embedder
from app.schemas import Hit


def dedup_records(
    records: list[Hit],
    embedder: Embedder,
    threshold: float | None = None,
) -> list[Hit]:
    """클러스터링 후 클러스터당 최신 답변만 유지. official은 그대로 보존."""
    threshold = threshold or settings.dedup_threshold

    official = [r for r in records if r.source == "official"]
    history = [r for r in records if r.source == "history"]
    if not history:
        return official

    print(f"  dedup: history={len(history)} (official {len(official)}는 보존)")

    print("  encoding question vectors for dedup...")
    vecs = embedder.encode_passages([r.question for r in history], show_progress=True)
    n, d = vecs.shape

    # rep_buf는 cluster representative 벡터를 누적. chunk 단위로 grow해서 재할당 비용 amortize.
    chunk = 4096
    rep_buf = np.empty((chunk, d), dtype=np.float32)
    n_reps = 0
    clusters: list[list[int]] = []

    for i in tqdm(range(n), desc="  clustering"):
        v = vecs[i]
        if n_reps > 0:
            sims = rep_buf[:n_reps] @ v  # normalized → cosine
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                clusters[best].append(i)
                continue
        if n_reps == rep_buf.shape[0]:
            rep_buf = np.concatenate([rep_buf, np.empty((chunk, d), dtype=np.float32)])
        rep_buf[n_reps] = v
        n_reps += 1
        clusters.append([i])

    deduped: list[Hit] = []
    for cluster in clusters:
        best_idx = max(cluster, key=lambda j: (_date_key(history[j].answered_at), -j))
        deduped.append(history[best_idx])

    print(
        f"  dedup result: {len(history)} → {len(deduped)} "
        f"({(1 - len(deduped) / len(history)) * 100:.1f}% reduction)"
    )
    return official + deduped


def _date_key(d: date | None) -> date:
    return d if d is not None else date(1900, 1, 1)
