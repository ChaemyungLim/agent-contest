"""의존성이 가벼운 텍스트/날짜 유틸. 단위 테스트가 ML 패키지 없이도 돌아야 함."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """경량 토크나이저.

    한국어/영문/숫자 토큰 단위 분리. 대소문자 무시. konlpy 미사용.
    BM25 + 키워드 라우팅 양쪽에서 동일 토크나이저를 사용해 일관성 보장.
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def parse_date(v: Any) -> date | None:
    """heterogeneous 입력을 date로 정규화. 실패 시 None.

    pandas Timestamp는 type name 검사로 다룬다 (이 모듈은 pandas에 의존하지 않음).
    """
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if type(v).__name__ == "Timestamp":  # pandas.Timestamp without importing pandas
        try:
            return v.to_pydatetime().date()
        except Exception:
            return None
    if isinstance(v, str):
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            pass
        try:
            from dateutil import parser as _dp
            return _dp.parse(v).date()
        except (ValueError, TypeError, OverflowError, ImportError):
            return None
    if isinstance(v, float):
        # NaN check
        return None
    return None
