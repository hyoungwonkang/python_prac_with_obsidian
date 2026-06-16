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

## 자동 동기 구축 계획 (B-1)

vault `.md` 파일을 commit하면 post-commit hook이 자동으로 Notion 미러를 갱신하는 구조. **결정일 2026-06-16**.

### 결정 사항

| 항목 | 선택 | 이유 |
|---|---|---|
| 동기 방식 | **B-1: Python script + Notion API** | 안정·무비용. Claude 세션 없이도 동작 (MCP 비의존). |
| 실습 코드 표시 | **GitHub 임베드/링크** | repo public이라 push만 하면 Notion이 항상 최신 코드 미리보기. |
| 트리거 | git `post-commit` hook | 별도 daemon·cron 없이 commit 시점에 즉시 |

### 사용자 작업 (Notion Integration Token 발급, 한 번만)

1. Notion → Settings → Connections → "Develop or manage integrations" (또는 https://www.notion.so/profile/integrations)
2. `+ New integration` → Name `vault-sync`, Type **Internal**, 워크스페이스 선택
3. Capabilities: `Read content`, `Update content`, `Insert content`. User capabilities: `No user information`.
4. 생성 후 **Internal Integration Secret** 복사 (`secret_...` 또는 `ntn_...` 형태)
5. 마스터 페이지 (page_id `38147515-0358-81f4-9b5f-c2fb013fa88a`)에서 `...` → Connections → `vault-sync` 추가 (자식 8개는 상속)
6. 토큰 저장 (vault 밖, mode 600):
   ```bash
   mkdir -p ~/.config/notion-sync
   echo 'YOUR_TOKEN' > ~/.config/notion-sync/token
   chmod 600 ~/.config/notion-sync/token
   ```

### 클로드 작업 (8단계)

1~5단계는 **2026-06-16 구현 완료** (`--dry-run` 으로 변환 검증: pytorch-study 75블록·to_do 21·GitHub 카드 7개, llm-from-scratch 83블록·to_do 50). 6·8단계는 **사용자 토큰 발급 후** 진행.

| # | 작업 | 산출물 | 상태 |
|---|---|---|---|
| 1 | `.notion-sync/` 디렉터리 + 매핑 yaml | `.notion-sync/config.yaml` | ✅ |
| 2 | 메인 동기 스크립트 | `sync.py` — git diff로 변경 `.md` 검출 → 변환 → 페이지 전량 교체 | ✅ |
| 3 | 마크다운 → Notion blocks 변환 | `converter.py` (mistune + `task_lists` 플러그인) | ✅ |
| 4 | GitHub embed 생성 | `github_link.py` — 실습 코드 파일명 검출 → blob URL → Notion bookmark 카드 | ✅ |
| 5 | 의존성·문서 | `requirements.txt`, `README.md` | ✅ |
| 6 | git hook 설치 | `.git/hooks/post-commit` (실행 권한 + `sync.py` 호출) | ⏳ 토큰 후 (`README.md`에 설치 명령) |
| 7 | 이 파일의 "운영 원칙" 갱신 | "수동" → "post-commit 자동, MCP는 보조" | ⏳ 6단계 검증 후 |
| 8 | 테스트 | vault 일부 수정 → commit → Notion 자동 반영 확인 | ⏳ 토큰 후 |

> **다음 액션 (사용자)**: 위 "사용자 작업" 절대로 `vault-sync` Integration Token 발급 →
> `~/.config/notion-sync/token` (mode 600) 저장. 그 후 `python .notion-sync/sync.py --all` 1회로
> 두 페이지 첫 미러 생성 → 정상 확인되면 6단계 hook 설치.
>
> 주의: 동기는 **페이지 전량 교체**(기존 children 삭제 후 재생성). Notion 직접 편집분은 덮어쓰임 — 미러 원칙대로.

### Notion 페이지 매핑 (현재 확보된 ID)

vault 노트가 있는 페이지만 동기 대상.

| vault | Notion page_id |
|---|---|
| `10-Projects/llm-from-scratch.md` | `38147515-0358-81f4-9b5f-c2fb013fa88a` (마스터) |
| `10-Projects/pytorch-study.md` | `38147515-0358-81e7-88be-edf4c2a76be2` (부록 A) |

본문 1~7장은 vault 노트 아직 없음 → 학습 시 `10-Projects/llm-ch{n}-*.md` 생성하면 매핑 추가:

| 장 | Notion page_id |
|---|---|
| 1장 | `38147515-0358-819e-8466-f57e31b34bbd` |
| 2장 | `38147515-0358-8138-95fb-e841ae4fb52a` |
| 3장 | `38147515-0358-815c-af30-c8728d23b5ce` |
| 4장 | `38147515-0358-81d8-b465-f9faab13fe3a` |
| 5장 | `38147515-0358-812e-a3c4-e1f404b0a6a0` |
| 6장 | `38147515-0358-8167-a267-ce0d17b844f1` |
| 7장 | `38147515-0358-817c-8e13-eae2e416690a` |

### Python 의존성

- `notion-client>=2.4.0` (공식 SDK)
- `pyyaml>=6.0`
- `mistune>=3.0` (마크다운 파서)

### 보안 원칙

- **토큰은 vault 밖** (`~/.config/notion-sync/token`, mode 600). git에 절대 안 들어감.
- `.gitignore`에 `.notion-sync/.cache/` 추가 (캐시·로그용).
- Integration 권한: content read/update/insert만, User info 없음 → 마스터 페이지 트리 밖 접근 불가.

## 관련 노트

- [[../10-Projects/pytorch-study]] — 첫 미러 대상 프로젝트
- [[../10-Projects/llm-from-scratch]] — Notion 마스터 페이지 정본
- [[../90-Daily/2026-06-16]] — 등록·인증·미러 생성 기록
- [[pytorch-env-hybrid]] — 유사 패턴: 외부 도구·환경 정본화
