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
| UI (Phase 1~3) | **Jinja2 템플릿** (확정) | FastAPI 단일 앱 안에서 SSR. 백엔드 학습에 집중하기 위한 단순 구조. |
| UI (Phase 4~) | **Next.js 분리** (확정) | API/프론트 분리 아키텍처 + React/TypeScript 학습. 사실상 새 프로젝트 수준 리팩토링. |
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

### Supabase 연결 방식: IPv4 Shared Pooler

- **Pooler (연결 풀러)**: DB 연결을 미리 만들어두고 재사용하는 중간 프록시 (PgBouncer 기반). 요청마다 연결을 새로 만들지 않아 빠르고 동시 접속에 강함.
- **Shared**: Supabase 무료 플랜에서 제공. 여러 사용자가 풀러 인프라를 공유. (유료 플랜은 Dedicated 전용 풀러 제공)
- **IPv4**: Supabase는 기본 IPv6인데, 로컬 환경이나 Vercel 등 IPv6 미지원 환경을 위해 IPv4 호환 엔드포인트를 별도 제공.
- 포트: Shared Pooler는 `6543`, Direct 연결은 `5432`
- 무료 플랜에서 안정적으로 접속하기 위한 권장 방식.

## UI 전환 전략 (Jinja → Next.js)

DB 교체와 달리 **아키텍처 자체가 바뀌는 큰 작업**. Phase 4는 사실상 새 프로젝트 수준.

| 측면 | Phase 1~3 (Jinja) | Phase 4~ (Next.js) |
|---|---|---|
| 구조 | FastAPI 단일 앱이 HTML 렌더링 | FastAPI=JSON API, Next.js=프론트 (분리) |
| 데이터 흐름 | form POST → 서버에서 처리 후 HTML 반환 | fetch/axios → JSON → 클라이언트에서 렌더링 |
| 폴더 | `app/templates/`, `app/static/` | `backend/`, `frontend/` (모노레포) |
| 배포 | Python 서버리스 함수 1개 | Python 함수 + Next.js 빌드 (Vercel 1급 지원) |
| 추가 학습 | (없음, Jinja만) | React, TypeScript, 라우팅, fetch, CORS |

**Phase 3에서 v1을 한 번 완성**한 뒤 Phase 4로 넘어가는 이유:
- 큰 리팩토링 도중 막혀도 동작하는 v1이 살아있음
- 분리 구조의 차이를 비교하며 학습할 수 있음 (같은 todo 앱의 두 가지 구현)

## 단계

### Phase 1: 로컬 학습 (FastAPI + Jinja + SQLite)

**1.A 백엔드 기초**
- [x] 1.1 FastAPI 프로젝트 셋업 (`requirements.txt`, `app/main.py`, 가상환경)
- [x] 1.2 기본 헬스체크 엔드포인트 (`GET /health`)
- [x] 1.3 Pydantic 모델 정의 (`TodoIn`, `TodoOut`, `TodoUpdate`)
- [x] 1.4 SQLAlchemy 비동기 엔진 + `Todo` ORM 모델
- [x] 1.5 Alembic 초기 설정 + 첫 마이그레이션
- [x] 1.6 CRUD 엔드포인트 — JSON API (`POST /api/todos`, `GET /api/todos`, ...)
- [x] 1.7 `/docs` Swagger UI에서 API 시나리오 검증

**1.B Jinja UI**
- [x] 1.8 `Jinja2Templates` 설정, `app/templates/` `app/static/` 디렉터리
- [x] 1.9 베이스 레이아웃 (`base.html`) + 정적 CSS
- [x] 1.10 목록 페이지 (`GET /` → `index.html`) — todo 표시
- [x] 1.11 추가 form (POST `/todos/add`) + 완료 토글 (POST `/todos/{id}/toggle`) + 삭제 (POST `/todos/{id}/delete`)
- [x] 1.12 브라우저에서 전체 흐름 확인

### Phase 2: Supabase 마이그레이션 (DB만 교체)

- [x] 2.1 Supabase 가입, 새 프로젝트 생성
- [x] 2.2 연결 정보 확인 (Connection string — pooler endpoint, IPv4)
- [x] 2.3 로컬에서 `DATABASE_URL`을 Supabase Postgres로 바꿔 Alembic 마이그레이션 적용
- [x] 2.4 ~~SQLite 데이터 이전~~ — 기존 데이터 0건, 생략
- [x] 2.5 로컬에서 Supabase 접속으로 CRUD + UI 동작 검증
- [x] 2.6 `.env`에 Supabase URL 보관, `.env.example` 작성

### Phase 3: Vercel 배포 v1 (Jinja 단일 앱)

- [x] 3.1 `vercel.json` 작성 (Python 함수 라우팅) + `api/index.py` 진입점 + `.vercelignore` + `lifespan`에서 `create_all` 제거 (Alembic 단일 소스)
- [ ] 3.2 Vercel 프로젝트 생성, GitHub 레포 연동
- [ ] 3.3 환경변수 등록 (`DATABASE_URL`, Supabase 키)
- [ ] 3.4 첫 배포 → Preview URL 확인
- [ ] 3.5 `/docs` + Jinja 페이지 배포 환경에서 CRUD 검증
- [ ] 3.6 데이터 영속성 검증 (함수 재시작 후 todo 유지)
- [ ] 3.7 **🏁 v1 마일스톤 — 동작하는 배포된 앱 완성**

### Phase 4: Next.js 분리 (대규모 리팩토링)

- [ ] 4.1 모노레포 구조 전환: `backend/`, `frontend/`로 폴더 분리
- [ ] 4.2 FastAPI를 **JSON API 전용**으로 다이어트 — Jinja·정적 파일 제거
- [ ] 4.3 CORS 설정 (`fastapi.middleware.cors`)
- [ ] 4.4 Next.js 프로젝트 생성 (`frontend/`, TypeScript)
- [ ] 4.5 todo 목록 페이지 (`app/page.tsx`) — fetch로 백엔드 API 호출
- [ ] 4.6 추가/수정/삭제 UI — 클라이언트 상태 관리
- [ ] 4.7 환경변수 (`NEXT_PUBLIC_API_BASE_URL`)로 백엔드 주소 분리
- [ ] 4.8 로컬에서 두 서버 동시 실행하며 검증

### Phase 5: Vercel 재배포 v2 (분리 구조)

- [ ] 5.1 `vercel.json` 또는 모노레포 설정 — 백엔드는 Python 함수, 프론트는 Next.js로 라우팅
- [ ] 5.2 환경변수 정비 (백엔드용 / 프론트용 분리)
- [ ] 5.3 배포 후 Preview URL에서 Next.js → FastAPI 흐름 검증
- [ ] 5.4 v1과 v2 비교 메모 → `[[30-References/jinja-vs-nextjs]]`로 학습 정리

## 검증 방법

| 단계 | 검증 |
|---|---|
| Phase 1 | 로컬 `uvicorn app.main:app --reload` → `http://localhost:8000/docs`에서 CRUD 5종 호출 성공. `http://localhost:8000/`에서 Jinja 페이지로 추가/토글/삭제 시나리오. 재시작 후 `todo.db`에서 데이터 유지. |
| Phase 2 | 로컬에서 `DATABASE_URL`만 Supabase로 바꾸고 Phase 1 시나리오 동일 통과. Supabase 대시보드 Table Editor에서 데이터 직접 보임. |
| Phase 3 | Vercel Preview URL의 `/docs` + 메인 페이지에서 CRUD 정상. 함수 재시작 후에도 데이터 유지. |
| Phase 4 | 로컬에서 `cd backend && uvicorn ...`, `cd frontend && npm run dev` 동시 기동. 브라우저(localhost:3000)에서 백엔드(localhost:8000) 호출하며 CRUD 정상. |
| Phase 5 | 배포된 Next.js URL에서 동일 시나리오. 네트워크 탭에 백엔드 API 호출이 보임. |

## 관련 노트

- `[[_Projects]]` (인덱스로 돌아가기)
- 스니펫·레퍼런스는 작업하면서 `[[20-Snippets/...]]`, `[[30-References/...]]`로 추가
