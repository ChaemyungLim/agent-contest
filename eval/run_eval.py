"""eval_set.jsonl 로 임계값 sweep + 지표 출력.

사용:
  python -m eval.run_eval --eval-set eval/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from app.retrieval.searcher import get_searcher
from app.schemas import Hit


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            items.append(json.loads(line))
    return items


def evaluate(items: list[dict], thresholds: Iterable[float]) -> None:
    searcher = get_searcher()
    # 캐시: 질문당 1회만 search
    cache: dict[str, list[Hit]] = {}
    for it in items:
        if it["query"] not in cache:
            cache[it["query"]] = searcher.search(it["query"])

    print(f"\n{'thresh':>8} {'P@1':>6} {'recall':>7} {'fb_acc':>7} {'F1':>6}")
    print("-" * 42)
    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for it in items:
            reranked = cache[it["query"]]
            top = reranked[0] if reranked else None
            answered = top is not None and top.score >= thresh
            expected = it["expected_match"]

            if expected and answered:
                if it.get("expected_doc_id") in (None, top.id):
                    tp += 1
                else:
                    fp += 1
            elif expected and not answered:
                fn += 1
            elif not expected and not answered:
                tn += 1
            else:  # not expected and answered
                fp += 1

        p_at_1 = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fb_acc = tn / (tn + fp) if tn + fp else 0.0
        f1 = (2 * p_at_1 * recall / (p_at_1 + recall)) if (p_at_1 + recall) else 0.0
        print(f"{thresh:>8.2f} {p_at_1:>6.3f} {recall:>7.3f} {fb_acc:>7.3f} {f1:>6.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, type=Path)
    ap.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="콤마 구분 임계값 리스트",
    )
    args = ap.parse_args(argv)

    items = load_eval_set(args.eval_set)
    if not items:
        print("eval set 비어있음", file=sys.stderr)
        return 1
    print(f"loaded {len(items)} eval items")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    evaluate(items, thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
