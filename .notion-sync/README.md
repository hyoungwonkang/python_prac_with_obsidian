# .notion-sync — vault → Notion 자동 미러

vault `.md` 노트를 commit 하면 그 변경을 Notion 미러 페이지에 자동 반영하는
동기 스크립트(B-1). **vault 가 단일 진실 공급원, Notion 은 미러.**

정본 규칙·설계: [`30-References/notion-mcp.md`](../30-References/notion-mcp.md)

## 구성

| 파일 | 역할 |
|---|---|
| `config.yaml` | vault 경로 → Notion `page_id` 매핑, github·토큰 경로 |
| `sync.py` | 메인 — 변경 `.md` 검출 → 변환 → 페이지 전량 교체 |
| `converter.py` | Markdown(mistune) → Notion blocks 변환 |
| `github_link.py` | 노트가 참조한 실습 코드 → GitHub bookmark 카드 |
| `requirements.txt` | `notion-client`, `pyyaml`, `mistune` |

> 동기 단위는 **페이지 전량 교체**(기존 children 삭제 후 재생성)라서 Notion 에서
> 직접 편집한 내용은 다음 동기 때 vault 기준으로 덮어쓴다 — 미러 원칙대로.

## 사용자 1회 설정 (토큰 발급)

스크립트가 동작하려면 Notion Internal Integration Token 이 필요하다.
(상세: `notion-mcp.md` "사용자 작업" 절)

1. https://www.notion.so/profile/integrations → **+ New integration**
   - Name `vault-sync`, Type **Internal**, 워크스페이스 선택
   - Capabilities: Read / Update / Insert content, **User: No user information**
2. 생성 후 **Internal Integration Secret** 복사 (`secret_...` / `ntn_...`)
3. 마스터 페이지(`38147515-0358-81f4-9b5f-c2fb013fa88a`)에서
   `...` → **Connections → vault-sync** 추가 (자식 페이지는 상속)
4. 토큰 저장 (vault 밖, mode 600):
   ```bash
   mkdir -p ~/.config/notion-sync
   printf '%s' 'YOUR_TOKEN' > ~/.config/notion-sync/token
   chmod 600 ~/.config/notion-sync/token
   ```
   또는 환경변수 `NOTION_TOKEN` 으로 줘도 된다(`token_path` 보다 우선).

## 설치

```bash
python -m pip install -r .notion-sync/requirements.txt
```

## 실행

```bash
# git diff HEAD~1..HEAD 의 변경 .md 동기 (post-commit 기본 동작)
python .notion-sync/sync.py

# 매핑된 모든 페이지 강제 동기
python .notion-sync/sync.py --all

# 특정 파일만
python .notion-sync/sync.py 10-Projects/pytorch-study.md

# API 호출 없이 변환 결과만 점검 (토큰 불필요)
python .notion-sync/sync.py --all --dry-run
```

## post-commit hook 설치 (자동 트리거)

`.git/hooks` 는 git 에 포함되지 않으므로 PC 마다 한 번 설치한다.

```bash
cat > .git/hooks/post-commit <<'SH'
#!/bin/sh
# vault .md commit 시 Notion 미러 자동 동기 (토큰 없으면 조용히 skip)
ROOT="$(git rev-parse --show-toplevel)"
VENV="$ROOT/.notion-sync/.cache/venv/bin/python"
if [ -x "$VENV" ]; then PY="$VENV"
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi
"$PY" "$ROOT/.notion-sync/sync.py" || true
SH
chmod +x .git/hooks/post-commit
```

> 인터프리터 우선순위: `.notion-sync/.cache/venv`(deps 설치된 venv) → `python3` → `python`.
> 시스템 Python 에 `notion-client` 등이 없으면 venv 를 먼저 만들어 둔다:
> ```bash
> uv venv .notion-sync/.cache/venv      # 또는 python3 -m venv
> uv pip install --python .notion-sync/.cache/venv -r .notion-sync/requirements.txt
> ```

> 토큰이 없거나 동기 대상이 없으면 스크립트가 안전하게 빠져나오므로
> 일반 커밋 흐름을 막지 않는다.

## 새 페이지 매핑 추가

본문 1~7장 노트(`10-Projects/llm-from-scratch/llm-ch{n}-*.md`)를 만들면
`config.yaml` `pages:` 의 주석 처리된 매핑을 해제한다. page_id 목록은
`notion-mcp.md` "Notion 페이지 매핑" 표가 정본.

## 보안

- 토큰은 **vault 밖**(`~/.config/notion-sync/token`)에만. git 에 절대 들어가지 않음.
- `.notion-sync/.cache/` 는 `.gitignore` 처리(캐시·로그용).
- Integration 권한은 content read/update/insert + User info 없음 → 마스터 페이지 트리 밖 접근 불가.
