# todo-app

할일관리 웹앱. 파이썬 중급 학습용 첫 프로젝트.

## Context

- 파이썬으로 중급 수준 예제 프로젝트를 연습.
- 최종 목표: GitHub에 올리고 **Vercel로 배포**.

## 기술 스택

| 구분 | 선택 | 이유 |
|---|---|---|
| 백엔드 프레임워크 | **FastAPI** (확정) | Vercel 공식 파이썬 예제가 FastAPI 기반 → 배포 마찰 적음. 타입힌트·Pydantic·async가 "중급 학습" 목표와 부합. |
| 데이터 저장 | _미정_ | SQLite(로컬) / Vercel Postgres / Supabase 후보 |
| UI | _미정_ | Jinja 템플릿 / 정적 HTML+JS / Next.js 후보 |
| 배포 | Vercel + GitHub | 자동 배포 파이프라인 |

> ⚠️ Vercel 제약:
> - 영구 파일시스템 없음 → 파일 기반 JSON/SQLite 저장은 **로컬에서만** 작동, 배포 시 외부 DB 필요
> - 함수당 실행시간 10초 (Hobby 플랜)
> - Streamlit·tkinter 같은 상시 GUI 앱 부적합

## 단계 (예정)

> DB·UI 결정 후 채워넣을 예정.

- [ ] Step 1: FastAPI 프로젝트 셋업, 기본 헬스체크 엔드포인트
- [ ] Step 2: Pydantic 모델 + CRUD 엔드포인트 (인메모리)
- [ ] Step 3: 영속 저장소 연결 (DB 결정 후)
- [ ] Step 4: UI (UI 결정 후)
- [ ] Step 5: GitHub 푸시 + Vercel 연동 + 환경변수 설정
- [ ] Step 6: 배포 검증

## 검증 방법

- 로컬: `uvicorn main:app --reload` → `/docs` Swagger UI에서 CRUD 호출
- 배포: Vercel preview URL에서 동일한 시나리오 동작
- 데이터 영속성: 함수 재시작 후에도 todo가 유지되는지 (외부 DB 검증)

## 관련 노트

- `[[_Projects]]` (인덱스로 돌아가기)
- 스니펫·레퍼런스는 작업하면서 `[[20-Snippets/...]]`, `[[30-References/...]]`로 추가
