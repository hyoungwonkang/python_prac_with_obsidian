# todo-app

할일관리 웹앱. 파이썬 중급 학습용 첫 프로젝트.

## Context

- 파이썬으로 중급 수준 예제 프로젝트를 연습.
- 최종 목표: GitHub에 올리고 **Vercel로 배포**.

## 기술 스택

| 구분 | 선택 | 이유 |
|---|---|---|
| 백엔드 프레임워크 | **FastAPI** (확정) | Vercel 공식 파이썬 예제가 FastAPI 기반 → 배포 마찰 적음. 타입힌트·Pydantic·async가 "중급 학습" 목표와 부합. |
| 데이터 저장 (Phase 1) | **SQLite** (확정) | 표준 라이브러리 내장, 외부 의존 0. SQL·ORM 개념 학습에 집중. |
| 데이터 저장 (Phase 2) | **Supabase** (확정) | Postgres 호스팅 + 풍부한 무료 티어(500MB, 5GB 전송) + 추후 Auth/Storage 확장 가능. Vercel 비종속이라 호스팅 이전 자유. |
| UI | _미정_ | Jinja 템플릿 / 정적 HTML+JS / Next.js 후보 |
| 배포 | Vercel + GitHub | 자동 배포 파이프라인 |

> ⚠️ Vercel 제약:
> - 영구 파일시스템 없음 → SQLite 파일은 **로컬에서만** 작동. 배포 시 Supabase로 교체.
> - 함수당 실행시간 10초 (Hobby 플랜)
> - Streamlit·tkinter 같은 상시 GUI 앱 부적합

## 데이터 저장 전환 전략 (SQLite → Supabase)

마이그레이션 비용을 줄이기 위해 처음부터 **DB 추상화 계층**을 거쳐 접근:

- **SQLAlchemy ORM 사용** — 모델 정의는 DB 무관. 연결 URL만 바꾸면 SQLite ↔ Postgres 전환됨.
- **`DATABASE_URL` 환경변수**로 분기:
  - 로컬: `sqlite+aiosqlite:///./todo.db`
  - 배포: `postgresql+asyncpg://...@aws-0-...supabase.com:5432/postgres`
- **마이그레이션 도구**: Alembic — 스키마 변경을 SQLite·Postgres 양쪽에 적용 가능
- **타입 호환성 주의**: SQLite엔 없는 Postgres 전용 타입(예: `JSONB`, `ARRAY`) 회피

→ 결과적으로 Phase 2 마이그레이션 작업은 **연결 URL 교체 + Alembic 스키마 적용 + 데이터 이전**으로 축소됨.

## 단계

### Phase 1: 로컬 학습 (SQLite)

- [ ] 1.1 FastAPI 프로젝트 셋업 (`requirements.txt`, `main.py`, 가상환경)
- [ ] 1.2 기본 헬스체크 엔드포인트 (`GET /health`)
- [ ] 1.3 Pydantic 모델 정의 (`TodoIn`, `TodoOut`, `TodoUpdate`)
- [ ] 1.4 SQLAlchemy 비동기 엔진 + `Todo` ORM 모델
- [ ] 1.5 Alembic 초기 설정 + 첫 마이그레이션
- [ ] 1.6 CRUD 엔드포인트 (`POST /todos`, `GET /todos`, `GET /todos/{id}`, `PATCH /todos/{id}`, `DELETE /todos/{id}`)
- [ ] 1.7 `/docs` Swagger UI에서 전체 시나리오 검증
- [ ] 1.8 UI 결정 후 프론트엔드 단계 추가 → `[[Phase 1 UI]]` 섹션 채우기

### Phase 2: Supabase 마이그레이션

- [ ] 2.1 Supabase 가입, 새 프로젝트 생성
- [ ] 2.2 연결 정보 확인 (Connection string, anon key)
- [ ] 2.3 로컬에서 `DATABASE_URL`을 Supabase Postgres로 바꿔 Alembic 마이그레이션 적용
- [ ] 2.4 SQLite의 기존 데이터 더미 export → Supabase로 import (선택, 학습용 데이터라면 생략 가능)
- [ ] 2.5 로컬에서 Supabase 접속으로 CRUD 검증

### Phase 3: Vercel 배포

- [ ] 3.1 `vercel.json` 작성 (Python 함수 라우팅)
- [ ] 3.2 Vercel 프로젝트 생성, GitHub 레포 연동
- [ ] 3.3 환경변수 등록 (`DATABASE_URL`, Supabase 키)
- [ ] 3.4 첫 배포 → Preview URL 확인
- [ ] 3.5 `/docs` 배포 환경에서 CRUD 검증
- [ ] 3.6 데이터 영속성 검증 (함수 재시작 후 todo 유지되는지)

## 검증 방법

| 단계 | 검증 |
|---|---|
| Phase 1 | 로컬 `uvicorn main:app --reload` → `http://localhost:8000/docs`에서 CRUD 5종 호출 성공. 재시작 후 `todo.db`에서 데이터 유지 확인. |
| Phase 2 | 로컬에서 `DATABASE_URL`만 Supabase로 바꾸고 동일 테스트 통과. Supabase 대시보드 Table Editor에서 데이터 직접 보임. |
| Phase 3 | Vercel Preview URL의 `/docs`에서 CRUD 정상. 함수 재시작(시간차 또는 강제) 후에도 데이터 유지. |

## 관련 노트

- `[[_Projects]]` (인덱스로 돌아가기)
- 스니펫·레퍼런스는 작업하면서 `[[20-Snippets/...]]`, `[[30-References/...]]`로 추가
