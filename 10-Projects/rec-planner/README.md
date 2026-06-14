# rec-planner

자연어 요청 → **검증된 추천 플랜**(일자별 활동 일정)을 만드는 경량 LLM 앱.
[[../../30-References/HARNESS]] 가이드의 6대 하네스 골격을 실제로 적용한 학습 프로젝트.

> 철학: **엔진(LLM)의 출력을 절대 신뢰하지 않는다. 검증층이 보정한다.**

## 구조

```
rec-planner/
├── app/
│   ├── main.py              # FastAPI 앱 + 예외→상태코드 매핑: 검열 400 / 파싱 422 (골격 6)
│   └── routers/plan.py      # 얇은 어댑터 (PlanRequest → pipeline → PlanResponse)
├── core/
│   ├── schema.py            # 핸드오프 계약 PlanRequest/Intent/Plan (골격 2)
│   ├── catalog.py           # 로컬 catalog.json 로더 (ES/SKU DB 대체)
│   ├── guardrail.py         # 입력 검열 워커 + GuardrailBlockedError (골격 4·6)
│   ├── pipeline.py          # ⭐ Step 0~4 + verify→fix 루프 (골격 1·5, LLM 주입식)
│   ├── validator.py         # ⭐ 검증 루프: parse 3전략 + 환각 제거 + total 재계산 (골격 5·6)
│   └── llm.py               # Haiku 의도분석 / Opus 플랜생성 (골격 4)
├── data/catalog.json        # 추천 후보 = 환각 검증의 정답지
├── tests/                   # 의존성 0 테스트 21개 (validator·guardrail·pipeline)
├── requirements.txt
└── .env.example             # ANTHROPIC_API_KEY
```

## 실행

```bash
# 1) 의존성
pip install -r requirements.txt

# 2) 검증 루프 테스트 (anthropic 불필요)
python -m pytest tests/test_validator.py -v

# 3) 서버 (ANTHROPIC_API_KEY 필요)
cp .env.example .env   # 키 입력
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
# → http://localhost:8000/docs 에서 POST /plan
```

## 요청 예시

```json
POST /plan
{"query": "주말에 가족이랑 실내 위주로 저렴하게", "days": 2, "adults": 2, "children": 1, "budget": 50000}
```

`itemId`가 카탈로그에 없으면 응답에서 제거되고, `totalPrice`는 항상 카탈로그 기준으로 재계산됩니다.
LLM이 제약(일수/예산)을 어기면 최대 3회까지 교정 재시도하고, 끝내 실패하면 결정적 폴백 플랜을 반환합니다.

## 레퍼런스

travel-rag (`/home/user1/final-project/mzc-final-project-be/services/travel-rag/`) —
특히 `recommend/validator.py`, `app/routers/recommend.py`와 1:1 대조.
