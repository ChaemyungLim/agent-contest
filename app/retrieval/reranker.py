"""bge-reranker-v2-m3 기반 cross-encoder 재정렬 (transformers 직접).

(query, passage) 페어를 입력해 단일 logit 출력. 값이 클수록 관련성 높음.
랭킹용이므로 sigmoid 미적용 (monotonic 보존).
"""
from collections.abc import Sequence
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import resolve_path, settings


class Reranker:
    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        path = str(resolve_path(model_path or settings.reranker_path))
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        )

    @torch.inference_mode()
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        enc = self.tokenizer(
            [query] * len(passages),
            list(passages),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits.view(-1).float()
        return logits.cpu().tolist()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
