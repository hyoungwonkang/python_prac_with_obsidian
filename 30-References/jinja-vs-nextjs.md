# Jinja (v1) vs Next.js 분리 (v2) — 같은 todo 앱 두 구현 비교

todo-app을 두 단계로 만들었음:
- **v1**: FastAPI + Jinja 단일 앱 (Phase 1~3)
- **v2**: FastAPI(JSON API) + Next.js(SSR/RSC) 분리 (Phase 4~5)

배포 URL (현재 시점):
- v1: ~~`https://python-prac-with-obsidian.vercel.app`~~ → v2 마이그레이션 후 backend(JSON only)로 전환됨
- v2 frontend: `https://todo-app-frontend-eight-lime.vercel.app`
- v2 backend (= 위 backend): `https://python-prac-with-obsidian.vercel.app`

## 아키텍처 한눈에

### v1 (단일 앱)
```
[브라우저]
   │ HTTP GET /
   ▼
[FastAPI 1개]
  ├─ Jinja templates → HTML 생성
  └─ /api/todos JSON API (보조)
   │
   ▼
[Supabase Postgres]
```

### v2 (분리)
```
[브라우저]
   │ GET /
   ▼
[Next.js Vercel]
  Server Component fetch
   │
   ▼
[FastAPI Vercel] ◄─── 브라우저 직접 호출 (CORS 통과)
  JSON API only
   │
   ▼
[Supabase Postgres]
```

## 파일·역할 비교

| 항목 | v1 | v2 |
|---|---|---|
| 백엔드 라우트 | `/`, `/todos/add`, `/api/todos`, ... | `/api/todos`만 (Jinja 라우트 제거) |
| 템플릿 | `app/templates/*.html` | 없음 |
| 정적 파일 | `app/static/style.css` (FastAPI 서빙) | Next.js public/ + Tailwind |
| 폴더 | `todo-app/` (단일) | `todo-app/backend/`, `todo-app/frontend/` |
| Vercel 프로젝트 | 1개 | 2개 (backend, frontend) |
| 환경변수 | `DATABASE_URL` | `DATABASE_URL`, `CORS_ORIGINS`, `NEXT_PUBLIC_API_BASE_URL` |
| 의존성 | Python만 (`requirements.txt`) | Python + Node (`requirements.txt` + `package.json`) |

## 요청 흐름 (한 todo 추가 시)

### v1
```
1. 브라우저: POST /todos/add (form data)
2. FastAPI: Form 파싱 → SQLAlchemy 저장
3. FastAPI: RedirectResponse("/", 303)
4. 브라우저: GET /
5. FastAPI: 목록 SELECT → Jinja 렌더 → HTML
6. 브라우저: 새 HTML로 페이지 전체 교체 (full reload)
```
요청 2회, 풀 리로드. 단순·전통적.

### v2
```
1. 브라우저: AddTodoForm 클릭 → fetch POST /api/todos (CORS)
2. FastAPI: JSON 파싱 → 저장 → JSON 응답
3. 브라우저: router.refresh() 호출
4. Next.js 서버: page.tsx 재실행 → fetch GET /api/todos
5. FastAPI: 목록 SELECT → JSON 응답
6. Next.js: 새 HTML+RSC payload 생성
7. 브라우저: 변경된 부분만 교체 (no full reload)
```
요청 더 많지만 풀 리로드 없음. 클라이언트가 영리해짐.

## 트레이드오프

| 측면 | v1 우세 | v2 우세 |
|---|---|---|
| 코드 단순함 | ✅ 한 언어, 한 프레임워크 | ⚠️ Python + TypeScript + Tailwind + Next.js |
| 초기 학습 곡선 | ✅ FastAPI 하나만 | ❌ 백/프론트 둘 다 |
| 배포 단순함 | ✅ Vercel 프로젝트 1개 | ⚠️ 2개 + CORS 설정 |
| UI 풍부함 | ❌ 풀 리로드 제약 | ✅ React 생태계 전체 |
| 상태 관리 | ❌ 서버 세션·쿠키 필요 | ✅ 클라이언트 상태 자연스러움 |
| 모바일 앱 재사용 | ❌ HTML만 반환 | ✅ JSON API 재사용 가능 |
| SEO | ✅ 서버 HTML | ✅ Next.js SSR도 동일 수준 |
| 첫 paint 속도 | ✅ 단일 요청 | ✅ SSR로 동일 수준 (RSC 덕분에 JS 작음) |
| 팀 분업 | ❌ 한 코드베이스 충돌 | ✅ 백/프론트 독립 |
| 디버깅 | ✅ 한 서버 로그 | ⚠️ 두 서버 + CORS + 환경변수 |
| 변경 영향도 | ⚠️ UI 바꾸려면 백엔드 수정 | ✅ 프론트만 수정해도 됨 |

## 언제 v1 vs v2?

v1 (Jinja 단일) 추천:
- 백엔드 개발자 혼자, UI는 부차적
- 관리 도구·내부 대시보드
- 풀 리로드가 UX에 OK인 단순 form 위주 앱
- 클라이언트 JS를 최소화하고 싶음
- 모바일 앱 계획 없음
- 한 사람이 빠르게 만들고 끝낼 프로젝트

v2 (분리) 추천:
- 풍부한 인터랙션·SPA 느낌이 중요
- 모바일 앱·다른 클라이언트도 같은 API 쓸 예정
- 백·프론트 분업
- SEO + 동적 UI 둘 다 필요
- 디자인 시스템·재사용 컴포넌트 적극 활용
- 장기 운영·확장 예상

## 학습 관점에서 얻은 것

이 프로젝트로 양쪽을 다 만들어본 결과:

### v1에서 배운 것
- FastAPI 라우팅·Form·RedirectResponse 패턴
- Jinja 템플릿 문법 (`{% %}`, `{{ }}`)
- SSR이 무엇인지 (서버에서 HTML 직접 생성)
- 단일 앱의 단순함이 주는 장점

### v2에서 배운 것
- Next.js App Router 구조 (`app/page.tsx`, `layout.tsx`)
- **Server Component vs Client Component** (`'use client'` 경계)
- **RSC가 SSR 위에 얹히는 방식** (JS 번들 축소)
- CORS의 실제 동작 (preflight, allow-origin, allow-credentials)
- 환경변수 분기 (`NEXT_PUBLIC_` 접두사 의미)
- Vercel의 모노레포 패턴 (한 레포 → 여러 프로젝트, Root Directory 분기)
- `router.refresh()`로 Server Component 재페치
- `useTransition`으로 비동기 작업 pending 처리
- Tailwind 유틸리티 클래스 기초

### 두 구현을 비교하며 느낀 것
- 같은 기능을 두 가지 방식으로 만들어보는 게 **트레이드오프 감각**에 결정적
- v1을 먼저 만들고 v2로 옮긴 게 정답 — 한번에 v2로 갔으면 "왜 분리하는지" 체감 못함
- 단순한 todo 앱 정도면 v1으로도 충분 — 분리는 미래 확장을 위한 투자
- Vercel 같은 PaaS는 v2 패턴을 1급으로 지원해서 분리 비용이 생각보다 낮음

## 관련 노트

- [[_References]] (인덱스로 돌아가기)
- [[../10-Projects/todo-app]] — 단계별 작업 내역
- [[vercel-project-switch]] — 다음 학습 프로젝트로 같은 Vercel 프로젝트 덮어쓰는 절차
- [[python-basics]] — Python·FastAPI 기초 용어
