# rec-planner

자연어 요청을 받아 **검증된 추천 플랜**(일자별 활동 일정)을 만드는 LLM 앱.
파이썬 중급 학습용 두 번째 프로젝트 — [[../30-References/HARNESS|HARNESS 가이드]]의 6대 골격을 실제로 적용한다.

## Context

- [[todo-app]](완료) 다음 단계: **LLM 하네스 엔지니어링** 학습.
- 레퍼런스 구현 `travel-rag`(여행 추천 RAG)를 `파일:줄` 단위로 대조하며 따라간다.
  - 위치: `/home/user1/final-project/mzc-final-project-be/services/travel-rag/`
  - 그 안의 원본 `HARNESS.md`가 정본, vault의 [[../30-References/HARNESS]]는 복사본.
- travel-rag의 무거운 인프라(ES·Kafka·Bedrock)를 떼어낸 **경량 버전** — 인프라 0, 하네스 80%(검증 루프 + 장애 처리)는 유지.

> 정체성: **travel-rag의 축소판 서비스**. HARNESS.md는 이 축소판을 짓기 위한 *설계서*, 하네스 엔지니어링은 그 설계서를 따라 짓는 *행위*. (도메인=추천, 방법=하네스)

## travel-rag 대비 축소 결정

| 측면 | travel-rag | rec-planner | 비고 |
|---|---|---|---|
| 도메인 | 여행(숙소·교통·활동, 박수×숙박가) | **당일 활동 추천** (서울 나들이) | 의식적 단순화 — 여행 아님 |
| 검색(RAG의 R) | 임베딩 + ES 하이브리드 검색 | **카테고리 필터** | R 생략, 학습 초점은 검증 루프 |
| 플랜 수 | 다중 티어(플랜 3개) | 단일 플랜 | 축소 |
| 검증 루프·intent·폴백 | — | **그대로 유지** | 충실 (하네스 핵심) |

## 핵심 철학

> 엔진(LLM)의 출력을 절대 신뢰하지 않는다. 검증층이 보정한다.

- LLM이 준 `itemId`가 실제 `catalog.json`에 있나? → 없으면 제거 (환각 방지)
- LLM이 계산한 `totalPrice`? → 무시하고 코드가 재계산 (산술 오류 방지)
- LLM이 깬 JSON? → 3전략으로 복구 (코드펜스 → 직접 → 괄호 카운팅)

## 기술 스택

| 구분 | 선택 | 이유 |
|---|---|---|
| 백엔드 | **FastAPI** (확정) | async·Pydantic이 핸드오프 계약/검증과 자연스럽고, travel-rag 레퍼런스와 동일 스택이라 1:1 대조 학습 가능. |
| LLM (경량 워커) | **Claude Haiku 4.5** (`claude-haiku-4-5`) | 의도 분석 — 빠르고 싸고 결정적(`temperature=0.1`). |
| LLM (메인 워커) | **Claude Opus 4.8** (`claude-opus-4-8`) | 플랜 생성 — adaptive thinking + `effort`. (sampling 파라미터 제거됨) |
| SDK | **anthropic** (Python) | `client.messages.create(...)`. Bedrock 대체. |
| "DB" | **로컬 `data/catalog.json`** | ES/SKU DB 대체. 환각 검증의 정답지. |
| 테스트 | pytest | 검증 루프부터 테스트(부록 A: 가장 먼저 만들 것). |

## 골격 매핑 (HARNESS 6대 ↔ 이 프로젝트)

| HARNESS 골격 | 파일 | 유지? |
|---|---|---|
| 1. 제어 흐름 (Step N 파이프라인) | `app/routers/plan.py` | ✅ Step 1~4 |
| 2. 핸드오프 계약 (타입) | `core/schema.py` (`PlanRequest`/`Intent`/`Plan`) | ✅ Pydantic |
| 3. 상태 영속화 | — | ❌ 생략 (동기 단일 요청) |
| 4. 워커 분담 | `core/llm.py` (Haiku 분류 ↔ Opus 생성) | ✅ |
| 5. 검증 루프 ⭐ | `core/validator.py` + `core/catalog.py` | ✅ 핵심 |
| 6. 장애 처리 | `validator.parse_llm_response` 3전략 + `app/main.py` 예외 매핑 | ✅ |

## 단계 (Phase)

### Phase 0: 경량 골격 셋업 — ✅ 완료 (검증 루프 테스트 12/12 통과)
- [x] 0.1 디렉터리 구조 + `requirements.txt` + `.env.example`
- [x] 0.2 `core/schema.py` — 핸드오프 계약 (Pydantic)
- [x] 0.3 `data/catalog.json` + `core/catalog.py` — 환각 검증 정답지
- [x] 0.4 ⭐ `core/validator.py` — 검증 루프 본체 (parse 3전략 + 환각 제거 + total 재계산)
- [x] 0.5 ⭐ `tests/test_validator.py` — 검증 루프 단위 테스트 (의존성 0, 가장 먼저 통과시킬 것)

### Phase 1: 워커 + 파이프라인 — 🛠 골격 작성 완료 (런타임 검증은 Phase 2.1)
- [x] 1.1 `core/llm.py` — Haiku 의도 분석 + Opus 플랜 생성 (+ 각 폴백)
- [x] 1.2 `app/routers/plan.py` — Step 1~4 파이프라인 + verify→fix 루프(`max_attempts=3`) + 결정적 폴백
- [x] 1.3 `app/main.py` — FastAPI 앱 + 예외→상태코드 매핑
  - ⚠️ 미설치 환경이라 실 LLM 왕복은 미검증 — 컴파일 OK, 폴백 로직 단독 검증 OK

### Phase 2: 검증 + 배포 (추후)
- [ ] 2.1 로컬 `uvicorn`으로 `/plan` 시나리오 검증
- [ ] 2.2 (선택) Vercel 배포 — todo-app 패턴 재사용

## 검증 방법

> **완료 게이트 (필수)**: 어떤 Phase를 "완료/done"으로 선언하기 전에 `[[../30-References/HARNESS]]`의 **6대 골격 `✅ 체크리스트`로 대조 감사**한다. 미충족 항목은 갭을 닫거나 *의식적 생략*으로 아래 현황에 기록한다. (이 규칙은 이 프로젝트 한정 — CLAUDE.md 전역 규칙으로 올리지 않음)

| 단계 | 검증 |
|---|---|
| Phase 0 | `pytest tests/test_validator.py` 통과 (anthropic 미설치여도 OK). |
| Phase 1 | `ANTHROPIC_API_KEY` 설정 후 `uvicorn app.main:app --reload` → `/docs`에서 `POST /plan` 호출. 환각 itemId가 응답에서 제거되고 totalPrice가 카탈로그 기준으로 재계산되는지 확인. |

### HARNESS 체크리스트 대조 현황 (최근 감사)

| 골격 | 상태 | 미충족/비고 |
|---|---|---|
| 1. 제어 흐름 | ✅ 충족 | Step 1~4 주석·산출물 전달·로그 |
| 2. 핸드오프 | ✅ 충족 | 병합 규칙은 단일 출처라 N/A |
| 3. 상태 영속화 | ➖ 의식적 생략 | 동기 단일 요청 (축소 결정) |
| 4. 워커 분담 | ⚠️ 일부 | 모델 분담·temperature ✅ / **guardrail 워커 미구현** |
| 5. 검증 루프 | ✅ 충족·테스트됨 | ID검증·재계산·상한·폴백·cap 전부 |
| 6. 장애 처리 | ⚠️ 일부 | 다전략 파싱·폴백 ✅ / **API 경계 매핑 얇음(guardrail 400·검증실패 422 실작동 X)** |

→ 남은 갭: 골격 4(guardrail 워커) + 골격 6(경계 매핑). 닫거나 "축소판 범위 밖"으로 *의식적 생략* 결정 필요.

## 관련 노트

- `[[_Projects]]` (인덱스)
- `[[../30-References/HARNESS]]` — 적용 중인 하네스 가이드
- 레퍼런스: travel-rag `recommend/validator.py`, `app/routers/recommend.py`
