"""사내 폐쇄망에서 bge-reranker-v2-m3 cross-encoder 동작 검증.

사용법:
    python verify_reranker.py [MODEL_DIR]

기본 경로: ./models/bge-reranker-v2-m3
필요 패키지: transformers>=4.40, torch>=2.1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


# 의미 관련성 sanity check 페어
QUERY = "환급율 조회 방법"
RELATED = "환급율은 그룹웨어 [장기인수>환급율조회] 메뉴에서 계약번호로 확인합니다."
UNRELATED = "오늘 날씨가 좋아서 산책하기 좋습니다."


def main(model_dir: str = "models/bge-reranker-v2-m3") -> int:
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
    ]
    missing = [f for f in required if not (path / f).exists()]
    has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    if missing or not has_weights:
        if missing:
            print(f"  ❌ 누락 파일: {missing}")
        if not has_weights:
            print("  ❌ weights 누락: pytorch_model.bin / model.safetensors 둘 다 없음")
        return 1
    print(f"  ✅ 필수 파일 {len(required)}개 + weights 존재")

    print("\n[2/5] 라이브러리 import")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        print(f"  ❌ import 실패: {e}")
        print("  → transformers / torch 설치 필요")
        return 1
    print(f"  ✅ torch {torch.__version__} / transformers OK")

    print("\n[3/5] tokenizer + 모델 로드")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return 1
    load_sec = time.time() - t0
    print(f"  ✅ 로드 완료 ({load_sec:.1f}초)")

    print("\n[4/5] 페어 점수 추론")
    t0 = time.time()
    pairs = [(QUERY, RELATED), (QUERY, UNRELATED)]
    with torch.inference_mode():
        enc = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        logits = model(**enc).logits.view(-1).float().cpu().tolist()
    infer_sec = time.time() - t0
    print(f"  ✅ {len(pairs)}개 페어 {infer_sec:.2f}초")
    print(f"     관련:  '{RELATED[:30]}...' → {logits[0]:+.3f}")
    print(f"     무관:  '{UNRELATED[:30]}...' → {logits[1]:+.3f}")

    print("\n[5/5] 정합성 검증")
    ok = True
    # logit 차이: 관련 페어가 무관 페어보다 점수 높아야
    diff = logits[0] - logits[1]
    if diff <= 0:
        print(f"  ❌ 관련성 역전: 관련({logits[0]:.3f}) ≤ 무관({logits[1]:.3f})")
        ok = False
    else:
        print(f"  ✅ 관련성 정렬: 관련 - 무관 = {diff:+.3f}")

    # logit 절대값 sanity (양극단으로 보이면 정상 동작)
    if abs(logits[0]) < 0.1 and abs(logits[1]) < 0.1:
        print("  ⚠️  logit이 0 근처로만 출력 — 모델 가중치 변형 의심")
        ok = False
    else:
        print(f"  ✅ logit 분포 정상 ({logits[0]:+.3f} / {logits[1]:+.3f})")

    print("\n" + ("=" * 40))
    if ok:
        print("🎉 검증 통과 — reranker 정상 동작")
        return 0
    print("❌ 검증 실패 — 위 ❌ 항목 확인 필요")
    return 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "models/bge-reranker-v2-m3"
    raise SystemExit(main(arg))
