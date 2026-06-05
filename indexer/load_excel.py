"""엑셀 로드 + 정규화.

입력 컬럼명은 환경변수(app/config.py)로 덮어쓸 수 있다.
official_faq 엑셀은 answered_at / answered_by가 없을 수 있다 → 선택적.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.config import settings
from app.retrieval.text import parse_date
from app.schemas import Hit, SourceType


def _normalize_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def load_excel(
    path: str | Path,
    source: SourceType,
    months_cutoff: int | None = None,
) -> Iterator[Hit]:
    """엑셀 한 파일을 읽어 정규화 Hit 시퀀스로 yield.

    Args:
        path: 엑셀 파일 경로
        source: "official" 또는 "history"
        months_cutoff: 시점 컷 (history에만 적용). None이면 비활성.
    """
    df = pd.read_excel(path, engine="openpyxl")

    q_col = settings.col_question
    a_col = settings.col_answer
    dt_col = settings.col_answered_at
    by_col = settings.col_answered_by

    if q_col not in df.columns or a_col not in df.columns:
        raise ValueError(
            f"엑셀 {path}에 필수 컬럼 ({q_col}, {a_col})이 없습니다. "
            f"실제 컬럼: {list(df.columns)}"
        )

    has_dt = dt_col in df.columns
    has_by = by_col in df.columns

    cutoff_date: date | None = None
    if months_cutoff and source == "history":
        cutoff_date = date.today() - timedelta(days=int(months_cutoff * 30.44))

    default_by = "장기인수팀" if source == "history" else "공식 FAQ"
    n_total = 0
    n_kept = 0

    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        n_total += 1
        row_dict = dict(zip(df.columns, row))
        question = _normalize_str(row_dict.get(q_col))
        answer = _normalize_str(row_dict.get(a_col))
        if not question or not answer:
            continue
        answered_at = parse_date(row_dict.get(dt_col)) if has_dt else None
        if cutoff_date and answered_at and answered_at < cutoff_date:
            continue
        answered_by = _normalize_str(row_dict.get(by_col)) if has_by else ""

        yield Hit(
            id=f"{source}-{idx}",
            question=question,
            answer=answer,
            source=source,
            answered_at=answered_at,
            answered_by=answered_by or default_by,
        )
        n_kept += 1

    print(f"  loaded {source}: {n_kept}/{n_total} rows (cutoff_date={cutoff_date})")
