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
3. **갱신 방향**: vault 노트 수정 → `git commit` → **post-commit hook이 `.notion-sync/sync.py` 자동 실행** → Notion 동기화. MCP는 보조. (2026-06-16부터 자동, 이전엔 수동.)
4. **충돌 시 vault 기준으로 갱신.** Notion에서 직접 체크박스 체크해도, 다음 동기 때 vault 기준으로 덮어씀.

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

### 사용자 작업 (Notion 연결 토큰 발급, 한 번만) — ✅ 2026-06-16 완료

> **용어 주의 (2026-06 Notion UI 변경)**: Notion이 개발자 화면에서 "integration(통합)"을
> **"connection(연결)"**으로 바꿨다. 그래서 아래 옛 안내의 "New integration"은 지금
> **"Create a new connection"(신규 연결 만들기)**이고, "Internal Integration Secret"은
> **"installation access token"**이다(역할 동일, `secret_`/`ntn_` 형태).

1. 개발자 포털 https://www.notion.so/my-integrations (→ `app.notion.com/developers/connections` 로 리다이렉트)
   왼쪽 **Build → Internal connections**
2. **Create a new connection** → Name `vault-sync`, 워크스페이스 선택 (새 연결은 기본 Internal)
3. Capabilities: `Read content`, `Update content`, `Insert content`. User: `No user information`.
4. **Configuration** 탭의 **installation access token** 복사
5. 마스터 페이지 (page_id `38147515-0358-81f4-9b5f-c2fb013fa88a`)에서 `•••` → **+ Add connections** → `vault-sync` 추가
   - ⚠️ **자식 8개는 마스터의 실제 하위 페이지일 때만 상속됨.** 트리 밖 별도 페이지면 각각 따로 공유 필요.
6. 토큰 저장 (vault 밖, mode 600):
   ```bash
   mkdir -p ~/.config/notion-sync
   printf '%s' 'YOUR_TOKEN' > ~/.config/notion-sync/token   # echo 대신 printf (개행 방지)
   chmod 600 ~/.config/notion-sync/token
   ```

### 클로드 작업 (8단계) — ✅ 2026-06-16 전 단계 완료

| # | 작업 | 산출물 | 상태 |
|---|---|---|---|
| 1 | `.notion-sync/` 디렉터리 + 매핑 yaml | `.notion-sync/config.yaml` | ✅ |
| 2 | 메인 동기 스크립트 | `sync.py` — git diff로 변경 `.md` 검출 → 변환 → 콘텐츠 블록 교체 | ✅ |
| 3 | 마크다운 → Notion blocks 변환 | `converter.py` (mistune + `task_lists`) | ✅ |
| 4 | GitHub embed 생성 | `github_link.py` — 실습 코드 → blob URL bookmark 카드 | ✅ |
| 5 | 의존성·문서 | `requirements.txt`, `README.md` | ✅ |
| 6 | git hook 설치 | `.git/hooks/post-commit` — 인터프리터 자동감지(venv→python3→python) | ✅ (per-PC, 미커밋) |
| 7 | 운영 원칙 갱신 | "수동" → "post-commit 자동, MCP는 보조" | ✅ (아래 "운영 원칙") |
| 8 | 실 동기 검증 | 마스터 91블록(83+child_page 8), pytorch-study 75블록, 토큰 정상 | ✅ |

#### 실 동기에서 잡은 버그 2개 (둘 다 수정 완료)

1. **하위 페이지 trash 사고** — "기존 children 전량 삭제" 전략이 마스터의 `child_page` 블록까지
   지워서 **자식 8개(부록A·1~7장)가 통째로 휴지통으로** 갔다. `pages.retrieve`는 되지만
   `blocks.children.list`가 `object_not_found` 나는 게 trash 시그니처. → `archived=False`로 8개 복구 +
   `sync.py`에 `PRESERVE_TYPES={child_page, child_database}` 추가(절대 삭제 안 함). **하위 페이지가 있는
   페이지를 미러할 땐 콘텐츠 블록만 교체하고 서브페이지는 보존**이 철칙.
2. **비-URL 링크 거부** — `[[todo-app]](완료)` 같은 위키링크+괄호가 mistune에 `[todo-app](완료)`
   마크다운 링크로 잡혀 url="완료"(한글)→ Notion `Invalid URL for link`. → converter가 **http/https/mailto
   절대 URL만 링크로 유지**, 상대경로·위키링크 잔재는 평문 처리.

> 동기는 **콘텐츠 블록만 전량 교체**(child_page/child_database 보존). Notion에서 직접 편집한 본문은
> 다음 동기 때 vault 기준으로 덮어쓰임 — 미러 원칙대로. 하위 페이지/DB는 안전.

### Notion 페이지 매핑 (현재 확보된 ID)

vault 노트가 있는 페이지만 동기 대상.

마스터(`llm-from-scratch.md`)는 **인덱스 = 체크박스 없음**, 각 장은 **자식 페이지**로 분리해 장별 정본 노트와 1:1 미러. (2026-06-16 분리 완료.)

| vault | Notion page_id | 비고 |
|---|---|---|
| `10-Projects/llm-from-scratch.md` | `38147515-0358-81f4-9b5f-c2fb013fa88a` | 마스터 (인덱스) |
| `10-Projects/pytorch-study.md` | `38147515-0358-81e7-88be-edf4c2a76be2` | 부록 A |
| `10-Projects/llm-from-scratch/llm-ch1-overview.md` | `38147515-0358-819e-8466-f57e31b34bbd` | 1장 |
| `10-Projects/llm-from-scratch/llm-ch2-text.md` | `38147515-0358-8138-95fb-e841ae4fb52a` | 2장 |
| `10-Projects/llm-from-scratch/llm-ch3-attention.md` | `38147515-0358-815c-af30-c8728d23b5ce` | 3장 |
| `10-Projects/llm-from-scratch/llm-ch4-gpt.md` | `38147515-0358-81d8-b465-f9faab13fe3a` | 4장 |
| `10-Projects/llm-from-scratch/llm-ch5-pretrain.md` | `38147515-0358-812e-a3c4-e1f404b0a6a0` | 5장 |
| `10-Projects/llm-from-scratch/llm-ch6-classify.md` | `38147515-0358-8167-a267-ce0d17b844f1` | 6장 |
| `10-Projects/llm-from-scratch/llm-ch7-instruct.md` | `38147515-0358-817c-8e13-eae2e416690a` | 7장 |
| `10-Projects/bert-classification.md` | `39247515-0358-80f6-839d-c52a09436dc5` | BERT 분류 마스터 (detection-ai-study 하위, 2026-07-03) |
| `10-Projects/bert-classification/bert-00-kickoff.md` | _(대기)_ | Phase 0 전환정리 — 마스터 자식 페이지 생성 후 매핑 |

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
