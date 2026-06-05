"""답변 채택 임계값 판정."""
from dataclasses import dataclass

from app.config import settings
from app.schemas import Hit


@dataclass
class Decision:
    answer: Hit | None
    fallback: bool
    alternatives: list[Hit]


def decide(reranked: list[Hit]) -> Decision:
    """리랭킹된 후보 리스트에서 답변 채택 여부 판정.

    상위 1건의 점수가 임계값 미달이면 부서 안내 fallback.
    """
    if not reranked:
        return Decision(answer=None, fallback=True, alternatives=[])

    top = reranked[0]
    if top.score < settings.answer_threshold:
        return Decision(answer=None, fallback=True, alternatives=reranked[:3])

    return Decision(answer=top, fallback=False, alternatives=reranked[1:3])
