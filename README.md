# 장기인수 FAQ RAG 챗봇

지점 매니저가 장기인수팀에 문의를 남기기 전, 과거 동일/유사 Q&A에서 답을
셀프서비스로 찾을 수 있게 해주는 1차 필터링 챗봇.

- **답변 소스**: 공식 FAQ (~300행) + 과거 매니저↔인수직원 Q&A (~30만행, 최근 12개월)
- **검색**: Dense(e5-base) + BM25 hybrid → RRF → bge-reranker-base
- **벡터 DB**: ChromaDB 임베디드 (별도 컨테이너 불필요)
- **답변 채택**: reranker 점수 ≥ 임계값(기본 0.5). 미달 시 부서 안내 fallback
- **부서 라우팅**: `app/routing/departments.yaml`의 키워드 매칭 (부서 2개)
- **LLM**: `app/llm/client.py` `call_llm()` 사내 LLM 연결 지점 (사용자 구현)

---

## 깃에 푸시하는 파일 (배포 자동화 입력)

```
.
├── Dockerfile           ← 컨테이너 정의
├── main.py              ← FastAPI 진입점 (uvicorn main:app)
├── requirements.txt
├── .env                 ← .env.example 참조 (gitignore)
├── .gitignore
├── app/                 ← 애플리케이션 모듈
├── indexer/             ← 인덱싱 CLI
└── eval/                ← 임계값 튜닝
```

`data/`, `models/`는 깃에 올리지 않음. 배포 환경에서 영속 볼륨/모델 레지스트리로 채움.

---

## 사전 준비

1. **모델**: 사내 모델 레지스트리에서 다음 경로로 마운트
   - `EMBEDDER_PATH` → `intfloat/multilingual-e5-base`
   - `RERANKER_PATH` → `BAAI/bge-reranker-base`

2. **엑셀**: `data/` 영속 볼륨에 배치
   ```
   data/official_faq.xlsx     # 컬럼: question, answer (선택: answered_at, answered_by)
   data/history_qa.xlsx       # 컬럼: question, answer, answered_at, answered_by
   ```
   컬럼명이 다르면 `.env`에 `COL_QUESTION` 등으로 덮어쓰기.

3. **부서 키워드**: `app/routing/departments.yaml`에 두 부서 담당자 회신 키워드 입력

4. **사내 LLM 연결**: `app/llm/client.py`의 `call_llm()` 구현 (MVP는 선택)

---

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# MVP: 300 공식 FAQ만
python -m indexer.build_index \
    --official data/official_faq.xlsx \
    --skip-history \
    --rebuild

# API 기동
uvicorn main:app --reload

# 테스트
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "신계약 고지의무 위반 시 처리 절차"}'
```

Phase 2 (30만 행 추가):
```bash
python -m indexer.build_index \
    --official data/official_faq.xlsx \
    --history data/history_qa.xlsx \
    --months 12 \
    --rebuild
```

---

## 배포

배포 인프라가 `Dockerfile`을 빌드해서 띄움. 컨테이너 시작 시 `main:app`이
ChromaDB 영속 디렉토리(`CHROMA_PERSIST_DIR`)와 BM25 인덱스 pkl을 로드한다.
**인덱스가 비어있으면 기동 실패**하므로, 첫 배포 후 인덱싱 잡을 한 번 실행:

```bash
# 배포된 컨테이너에서 또는 별도 잡으로
docker exec faq-chatbot python -m indexer.build_index \
    --official /app/data/official_faq.xlsx \
    --history /app/data/history_qa.xlsx \
    --months 12 \
    --rebuild
```

---

## 운영 중 튜닝

| 항목 | 환경변수 | 기본값 | 조정 시점 |
|---|---|---|---|
| 시점 컷 | `HISTORY_MONTHS` | 12 | 약관 개정 주기 따라 6/12/17 |
| 답변 임계값 | `ANSWER_THRESHOLD` | 0.5 | eval로 F1 최대 지점 |
| Rerank 후보 수 | `RERANK_TOP_K` | 20 | 응답 레이턴시 vs 정확도 |
| dedup 유사도 | `DEDUP_THRESHOLD` | 0.95 | 인덱싱 단계 |

임계값 튜닝:
```bash
# eval/eval_set.jsonl을 50~100건 라벨링 후
python -m eval.run_eval --eval-set eval/eval_set.jsonl
```

---

## 디렉토리

```
main.py                     # FastAPI 진입점
app/
├── api/chat.py             # POST /chat, GET /health
├── retrieval/              # 임베딩/리랭커/검색/임계값
├── llm/client.py           # 🔧 사내 LLM 연결 지점
├── routing/                # 부서 키워드 라우팅
├── config.py               # 환경변수 (pydantic-settings)
└── schemas.py              # 응답 모델
indexer/
├── load_excel.py
├── dedup.py
└── build_index.py          # 📥 인덱싱 CLI
eval/
├── eval_set.jsonl.example
└── run_eval.py
tests/                      # pytest 단위 테스트
```
