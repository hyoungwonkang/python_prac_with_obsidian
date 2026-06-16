---
title: Notion MCP 통합
tags: [reference, mcp, notion, tooling]
---

# Notion MCP 통합

Claude Code에서 Notion 워크스페이스를 직접 다루기 위한 MCP 서버 설정·운영 정본.

## 한 줄 요약

Notion 공식 호스티드 MCP를 HTTP/OAuth로 연결. **vault가 정본, Notion은 미러.** vault 노트가 바뀌면 Notion으로 동기, 반대 방향은 보조.

## 서버 등록 정보

| 항목 | 값 |
|---|---|
| 서버 이름 | `notion` |
| Transport | HTTP |
| URL | `https://mcp.notion.com/mcp` |
| 인증 | OAuth (브라우저) |
| 등록 스코프 | local (project: `/Users/apexesc/dev`) |
| 설정 파일 | `~/.claude.json` |

### 등록 명령

```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp
```

### 스코프 주의사항

local 스코프로 등록했기 때문에 **다른 cwd에서 `claude`를 시작하면 `notion`이 보이지 않는다.** PyTorch·vault 작업과 Notion을 같이 쓸 때는 `cd /Users/apexesc/dev` 후 `claude` 실행.

워크플로가 굳어지면 user 스코프로 옮기는 것 고려:

```bash
claude mcp remove notion
claude mcp add --scope user --transport http notion https://mcp.notion.com/mcp
```

## OAuth 인증 절차

1. `claude` 세션 안에서 `/mcp` 입력.
2. 목록에서 `notion` 선택 → `Authenticate`.
3. 브라우저가 열림 → Notion 계정 로그인 → 접근 권한을 줄 워크스페이스 선택.
4. 토큰은 MCP 클라이언트가 영구 저장 (재인증 불필요).
5. 검증: `claude mcp list`에서 `notion: ✓ Connected`.

### 접근 범위

- 인증 시 선택한 **워크스페이스 내 페이지·DB**만 접근 가능.
- 특정 페이지에 작업하려면 그 페이지가 인증 계정 트리에 있거나 통합에 공유돼야 함 (Notion 페이지 "공유 → 통합 추가").
- 노출되는 도구 이름은 `mcp__notion__*` 형태.

## 운영 원칙 (vault 정본 규칙과 정합)

1. **vault가 단일 진실 공급원.** `10-Projects/*.md`, `30-References/*.md`가 정본.
2. **Notion은 시각화·공유용 미러.** 페이지에 vault 경로를 callout으로 박아 출처 명시.
3. **갱신 방향**: vault 노트 수정 → `git commit` → Notion 동기화. 반대 방향은 보조.
4. **충돌 시 vault 기준으로 갱신.** Notion에서 직접 체크박스 체크해도, 다음 세션에서 vault 기준으로 다시 맞춤.

## 첫 페이지 설계 (pytorch-study 미러)

[[../10-Projects/pytorch-study]]를 Notion으로 미러링하는 첫 케이스.

### 구조

- **제목**: "PyTorch 학습 진행 — pytorch-study"
- **상단 callout**: 정본 출처 = `~/dev/python_prac_with_obsidian/10-Projects/pytorch-study.md`
- **본문 섹션**:
  1. Context / 기술 스택 (표)
  2. 환경 제약 요약 ([[pytorch-env-hybrid]] 참조)
  3. **Phase 0~5 체크리스트** (vault의 `- [ ]` / `- [x]`를 Notion to-do 블록으로)
  4. 작업 분담 원칙 (표)
  5. 활동 로그 — 첫 항목 = [[../90-Daily/2026-06-16]] 핵심 섹션
  6. 커밋 타임라인 — origin/main 7 커밋 (`94723c5`~`dc6f809`)
  7. 참고 링크 — vault 경로 + GitHub 경로 + Colab 노트북

### 필요한 입력

- Notion에서 사용자가 직접 만든 **부모 페이지 URL 또는 페이지명**.
- 부모 페이지를 통합(integration)에 공유해야 MCP가 접근 가능.

## 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| `! Needs authentication` 계속 표시 | `/mcp`에서 `notion` 재선택 후 `Authenticate` 다시. 워크스페이스 선택을 빠뜨리지 않았는지 확인. |
| `mcp__notion__*` 도구가 안 보임 | `claude` 재시작 안 한 상태 가능성. `/exit` 후 다시 실행. |
| `notion` 자체가 `claude mcp list`에 안 나옴 | local 스코프 → 다른 cwd. `cd /Users/apexesc/dev` 후 확인. |
| 특정 페이지 접근 거부 | 페이지가 통합 트리 밖. Notion 페이지 "공유 → 통합 추가". |

## 관련 노트

- [[../10-Projects/pytorch-study]] — 첫 미러 대상 프로젝트
- [[../90-Daily/2026-06-16]] — 등록·인증 대기 기록
- [[pytorch-env-hybrid]] — 유사 패턴: 외부 도구·환경 정본화
