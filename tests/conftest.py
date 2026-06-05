from datetime import date

import pytest

from app.schemas import Hit


@pytest.fixture
def make_hit():
    def _factory(
        id: str = "x-1",
        score: float = 0.0,
        source: str = "history",
        question: str = "q",
        answer: str = "a",
    ) -> Hit:
        return Hit(
            id=id,
            question=question,
            answer=answer,
            source=source,  # type: ignore[arg-type]
            answered_at=date(2025, 1, 1),
            score=score,
        )

    return _factory
