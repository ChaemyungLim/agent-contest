from pathlib import Path

import pytest
import yaml

from app.routing.department import load_router


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    data = {
        "departments": [
            {
                "id": "dept_a",
                "name": "신계약심사팀",
                "keywords": ["신계약", "고지의무", "예비심사"],
            },
            {
                "id": "dept_b",
                "name": "보험금심사팀",
                "keywords": ["보험금", "지급", "면책"],
            },
        ],
        "default": "dept_a",
    }
    p = tmp_path / "depts.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_keyword_match_routes_to_correct_dept(yaml_file: Path):
    router = load_router(yaml_file)
    dept = router.route("신계약 고지의무 위반 관련 문의입니다")
    assert dept.id == "dept_a"


def test_other_dept_keywords(yaml_file: Path):
    router = load_router(yaml_file)
    dept = router.route("보험금 지급 면책 사유 알려주세요")
    assert dept.id == "dept_b"


def test_no_match_returns_default(yaml_file: Path):
    router = load_router(yaml_file)
    dept = router.route("오늘 점심 뭐 먹지")
    assert dept.id == "dept_a"  # default


def test_tie_returns_default(yaml_file: Path):
    router = load_router(yaml_file)
    # dept_a 1점 (신계약), dept_b 1점 (보험금) → tie → default
    dept = router.route("신계약 보험금 관련 문의")
    assert dept.id == "dept_a"
