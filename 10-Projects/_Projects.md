# Projects 인덱스

각 프로젝트는 이 폴더에 `프로젝트명.md` 노트로 작성하고 아래에 wikilink로 등록.

## 진행 중

- [[rec-planner]] — 자연어 → 검증된 추천 플랜 LLM 앱. [[../30-References/HARNESS]] 6대 골격 적용(경량). 검증 루프(Phase 0) 완료, 워커·파이프라인(Phase 1) 골격 작성.

## 보류

- [[../99-Archive/contact-manager/contact|contact-manager-cli]] — 주소록 CLI (1단계 중 Step 1·2까지 완료, Obsidian 전환으로 보류)

## 완료

- [[todo-app]] — 할일관리 웹앱. Phase 1~5 전부 완료. v1(Jinja)과 v2(FastAPI+Next.js 분리) 두 구현 모두 Vercel 배포. 비교 회고: [[../30-References/jinja-vs-nextjs]]

## 프로젝트 규칙 (Conventions)

이 vault에서 새 파이썬 프로젝트를 시작할 때 따르는 규칙:

1. **`todo-app`은 FastAPI 고정** — Vercel 배포 호환성 + 중급 학습 목표.
2. **그 외 새 파이썬 웹 프로젝트 시작 시, Flask를 후보 프레임워크에 항상 포함** — 학습 자료가 풍부하고 서버사이드 렌더링·전통적 웹앱에 적합. FastAPI/Django 등과 함께 비교 검토.
3. **다른 PC에서 작업해도 같은 규칙이 유지되도록**, 위 결정은 이 노트(vault 정본)에 기록. (`~/.claude/...`의 메모리는 PC별 보조 캐시일 뿐.)

## 새 프로젝트 추가 방법

1. `10-Projects/` 폴더에 새 노트 생성: `프로젝트명.md`
2. 노트 상단에 Context · 목표 · 기술스택 · 단계별 할일 · 검증 방법 작성
3. 이 인덱스의 "진행 중" 섹션에 `[[프로젝트명]]` 추가
