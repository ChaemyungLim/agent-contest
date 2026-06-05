"""키워드 기반 부서 라우팅.

부서 단 2개를 가정. 각 부서가 자신의 업무 키워드를 제공한다.
질문 + 검색 결과 질문 텍스트에서 키워드 토큰 매칭 카운트로 라우팅.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import resolve_path, settings
from app.retrieval.text import tokenize
from app.schemas import DepartmentInfo, Hit


@dataclass
class Department:
    id: str
    name: str
    # 단일 토큰 키워드는 set 멤버십, 다중 토큰은 부분 시퀀스 일치
    keyword_tokens: set[str]
    keyword_phrases: list[list[str]]

    def to_info(self) -> DepartmentInfo:
        return DepartmentInfo(id=self.id, name=self.name)


@dataclass
class Router:
    departments: dict[str, Department]
    default_id: str

    def route(self, query: str, top_hits: list[Hit] | None = None) -> Department:
        text = " ".join([query, *(h.question for h in (top_hits or []))])
        tokens = tokenize(text)
        token_set = set(tokens)

        scores = {
            dept_id: (
                sum(1 for t in dept.keyword_tokens if t in token_set)
                + sum(1 for p in dept.keyword_phrases if _contains_sequence(tokens, p))
            )
            for dept_id, dept in self.departments.items()
        }
        top_score = max(scores.values())
        if top_score == 0:
            return self.departments[self.default_id]
        winners = [d for d, s in scores.items() if s == top_score]
        if len(winners) > 1:
            return self.departments[self.default_id]
        return self.departments[winners[0]]


def _contains_sequence(tokens: list[str], phrase: list[str]) -> bool:
    if not phrase:
        return False
    n = len(phrase)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == phrase:
            return True
    return False


def load_router(yaml_path: str | Path | None = None) -> Router:
    path = Path(yaml_path) if yaml_path else resolve_path(settings.departments_yaml)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    departments: dict[str, Department] = {}
    for entry in config["departments"]:
        keyword_tokens: set[str] = set()
        keyword_phrases: list[list[str]] = []
        for kw in entry.get("keywords", []):
            toks = tokenize(kw)
            if len(toks) == 1:
                keyword_tokens.add(toks[0])
            elif len(toks) > 1:
                keyword_phrases.append(toks)

        dept = Department(
            id=entry["id"],
            name=entry["name"],
            keyword_tokens=keyword_tokens,
            keyword_phrases=keyword_phrases,
        )
        departments[dept.id] = dept

    default_id = config.get("default") or next(iter(departments))
    if default_id not in departments:
        raise ValueError(f"default 부서 id '{default_id}'가 departments에 없습니다.")

    return Router(departments=departments, default_id=default_id)


@lru_cache(maxsize=1)
def get_router() -> Router:
    return load_router()


def reset_router() -> None:
    get_router.cache_clear()
