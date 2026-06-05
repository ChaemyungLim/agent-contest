"""사내 폐쇄망에서 bge-m3 임베딩 모델 동작 검증.

사용법:
    python verify_embedder.py [MODEL_DIR]

기본 경로: ./models/bge-m3
필요 패키지: transformers>=4.40, torch>=2.1, numpy
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


EXPECTED_DIM = 1024
TOLERANCE = 1e-5

SAMPLES = [
    "환급율 조회는 어떻게 하나요?",
    "담보삭제 신청 절차 알려주세요",
    "오늘 날씨가 좋네요",  # 도메인 무관 — 거리 비교용
]


def main(model_dir: str = "models/bge-m3") -> int:
    path = Path(model_dir).resolve()
    print(f"[1/5] 모델 경로 확인: {path}")
    if not path.exists():
        print(f"  ❌ 경로 없음: {path}")
        return 1
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "pytorch_model.bin",
    ]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        print(f"  ❌ 누락 파일: {missing}")
        return 1
    print(f"  ✅ 필수 파일 {len(required)}개 모두 존재")

    print("\n[2/5] 라이브러리 import")
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(f"  ❌ import 실패: {e}")
        print("  → transformers / torch / numpy 설치 필요")
        return 1
    print(f"  ✅ torch {torch.__version__} / transformers OK")

    print("\n[3/5] tokenizer + 모델 로드")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModel.from_pretrained(str(path)).eval()
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return 1
    load_sec = time.time() - t0
    print(f"  ✅ 로드 완료 ({load_sec:.1f}초)")

    print("\n[4/5] 임베딩 추출 (CLS pooling + L2 normalize)")
    t0 = time.time()
    with torch.inference_mode():
        enc = tokenizer(
            SAMPLES, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        cls = model(**enc).last_hidden_state[:, 0]
        vecs = F.normalize(cls, p=2, dim=1).cpu().numpy().astype(np.float32)
    infer_sec = time.time() - t0
    print(f"  ✅ {len(SAMPLES)}개 임베딩 {infer_sec:.2f}초")
    print(f"     shape: {vecs.shape}, dtype: {vecs.dtype}")

    print("\n[5/5] 정합성 검증")
    ok = True
    if vecs.shape != (len(SAMPLES), EXPECTED_DIM):
        print(f"  ❌ shape 불일치: {vecs.shape} != ({len(SAMPLES)}, {EXPECTED_DIM})")
        ok = False
    else:
        print(f"  ✅ shape OK ({len(SAMPLES)}, {EXPECTED_DIM})")

    norms = (vecs ** 2).sum(axis=1) ** 0.5
    if not np.allclose(norms, 1.0, atol=TOLERANCE):
        print(f"  ❌ L2 정규화 미적용: norms={norms}")
        ok = False
    else:
        print(f"  ✅ L2 정규화 OK (norms ≈ {norms.mean():.6f})")

    # 의미 유사도 sanity check (보험 질문 vs 무관 질문)
    sim_related = float((vecs[0] * vecs[1]).sum())     # 환급율 vs 담보삭제 (보험 도메인)
    sim_unrelated = float((vecs[0] * vecs[2]).sum())   # 환급율 vs 날씨
    if sim_related <= sim_unrelated:
        print(
            f"  ⚠️  의미 유사도 이상: 보험-보험({sim_related:.3f}) "
            f"≤ 보험-날씨({sim_unrelated:.3f})"
        )
        ok = False
    else:
        print(
            f"  ✅ 의미 유사도 OK: 보험-보험 {sim_related:.3f} "
            f"> 보험-날씨 {sim_unrelated:.3f}"
        )

    print("\n" + ("=" * 40))
    if ok:
        print("🎉 검증 통과 — 모델 정상 동작")
        return 0
    print("❌ 검증 실패 — 위 ❌ 항목 확인 필요")
    return 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "models/bge-m3"
    raise SystemExit(main(arg))
