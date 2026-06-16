# Project Guide for Claude Code

이 파일은 Claude Code가 세션 시작 시 자동으로 읽어들이는 프로젝트 가이드입니다.
git에 포함되어 있어 어느 PC에서 클론해도 동일한 규칙이 적용됩니다.

## Vault 개요

이 레포(`python_prac_with_obsidian`)는 **파이썬 학습용 Obsidian vault**.
경로는 PC별로 다르지만(예: macOS `~/dev/python_prac_with_obsidian/`, Windows `C:\dev\python-prac\`) **vault 구조와 규칙은 동일**하다.

- `10-Projects/` — 진행 중·보류·완료 프로젝트 단위 노트
- `20-Snippets/` — 모든 프로젝트가 공유하는 재사용 코드 조각
- `30-References/` — 외부 문서·링크
- `90-Daily/` — 데일리 노트
- `99-Archive/` — 보관

진행 중인 프로젝트와 vault 규칙(Conventions)은 `[[10-Projects/_Projects.md]]`를 참조.

## 작업 규칙 (Working Rules)

### 1. 의미있는 결정사항은 즉시 커밋 (질문 없이)

다음에 해당하는 결정·변경은 **사용자에게 커밋 여부를 묻지 말고 바로 `git commit`** 한다.
**Push는 별도로 사용자가 지시할 때만 수행** — 자동 push 금지.

**커밋 대상 (의미있는 결정)**
- 기술 스택 결정 변경 (framework, DB, ORM, 핵심 라이브러리)
- 프로젝트 시작·상태 전환 (진행 중 → 보류·완료, 새 프로젝트 노트 추가)
- vault 구조·규칙 변경 (폴더 추가/이동, Conventions 수정, CLAUDE.md 변경)
- Phase/Step 완료 표시 또는 명시적 마일스톤 도달
- 학습 결과로 정리된 스니펫·레퍼런스 추가

**커밋하지 않는 것 (작업 중 잡음)**
- 탐색 중인 임시 메모, 분석 중 메모
- 사용자가 "임시", "실험", "초안"이라고 말한 변경
- 명백한 오타 수정만 단독으로 (다음 의미있는 변경에 묶음)
- 되돌릴 가능성이 있는 시도

**커밋 메시지 컨벤션**
- Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:` 등
- 스코프 사용: `docs(todo-app): ...`, `feat(snippets): ...`
- 본문에는 **왜** 결정했는지 1~2줄 (vault 노트와 중복돼도 OK)
- 마지막 줄: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

**예시**
```
docs(todo-app): finalize data stack (SQLite → Supabase)

SQLite로 학습 시작 후 Supabase Postgres로 마이그레이션하는 2단계 전략.
SQLAlchemy ORM + DATABASE_URL 환경변수 분기로 마이그레이션 마찰 최소화.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 2. 메모리와 vault 노트의 역할 분리

- **vault 노트(`10-Projects/_Projects.md`, 각 프로젝트 노트, `CLAUDE.md`, `30-References/*`)** = git에 올라가는 **정본 (Single Source of Truth)**
- **메모리 (`<HOME>/.claude/projects/.../memory/`, PC별 다름)** = 이 PC 한정 보조 캐시
- **사용자 전역 `<HOME>/.claude/CLAUDE.md`** = vault를 가리키는 포인터일 뿐. 실제 결정은 vault 노트가 정본.
- 같은 결정을 양쪽에 둘 때, **vault 노트가 우선**. 메모리·전역 CLAUDE.md가 더 오래되면 vault 기준으로 갱신.

### 3. 프로젝트 규칙은 `_Projects.md`의 Conventions 섹션이 정본

기술 스택 후보·명명 규칙 등 프로젝트 콘텐츠 관련 규칙은 `[[10-Projects/_Projects.md]]`의 "프로젝트 규칙(Conventions)" 섹션이 정본.
이 CLAUDE.md는 Claude Code의 동작 규칙(메타)에 집중.

### 4. vault 노트에 명시된 결정은 재확인 없이 따른다

CLAUDE.md·`_Projects.md`·각 프로젝트 노트(예: `todo-app.md`)에 이미 결정·계획이 적혀 있으면 **다시 묻지 않고 그대로 진행**한다.

- 예: `todo-app.md`의 Phase 4가 "backend·frontend 분리"로 정의되어 있으면, "1개 유지 vs 2개 분리부터 정할까요?" 같은 재확인 질문 금지.
- 노트와 **충돌**하거나 노트에 **누락된 정보**가 있을 때만 질문.
- 변경이 필요하다고 판단되면 먼저 vault 노트 자체를 갱신하자고 제안 (그래야 다른 PC에서도 같은 결정 유지).

이유: 매번 묻는 행동은 vault가 정본이라는 [[#2-메모리와-vault-노트의-역할-분리|규칙 #2]]·다중 PC 일관성 철학과 모순. 같은 결정을 메모리·대화·vault 세 곳에서 반복 확인하면 마찰만 증가.

## Push 정책

- 자동 push 금지. 사용자가 명시적으로 "푸시" 또는 "push"라고 말할 때만 실행.
- 단일 결정의 작은 커밋이 여러 개 쌓여도 그대로 둠 (의미가 명확한 단위 유지).
