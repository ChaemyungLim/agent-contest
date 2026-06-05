"""RRF merge 로직 단위 테스트. searcher 모듈이 chromadb / rank_bm25에 의존하므로
미설치 시 자동 skip."""
import pytest

pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")

from app.retrieval.searcher import Searcher  # noqa: E402


def test_rrf_merge_promotes_doc_in_both_lists(make_hit):
    dense = [make_hit(id="a"), make_hit(id="b"), make_hit(id="c")]
    sparse = [make_hit(id="b"), make_hit(id="d")]

    fused = Searcher._rrf_merge(object(), dense, sparse)  # type: ignore[arg-type]
    ids = [h.id for h in fused]
    assert ids[0] == "b"  # 양쪽 등장 → 최상위
    assert set(ids) == {"a", "b", "c", "d"}
