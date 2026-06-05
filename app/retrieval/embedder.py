"""bge-m3 기반 임베딩 (transformers 직접).

dense embedding 명세:
  - CLS pooling (last_hidden_state[:, 0])
  - L2 normalize
  - 차원 1024
"""
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from app.config import resolve_path, settings


class Embedder:
    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        path = str(resolve_path(model_path or settings.embedder_path))
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()

    @torch.inference_mode()
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            vec = F.normalize(cls, p=2, dim=1)
            out.append(vec.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], batch_size=1)[0]

    def encode_passages(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self._encode(list(texts), batch_size or settings.embed_batch_size)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
