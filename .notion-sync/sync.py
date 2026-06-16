#!/usr/bin/env python3
"""sync.py — vault `.md` → Notion 미러 페이지 동기 (B-1 자동 동기 메인).

흐름:
1. 변경된 `.md` 검출 (post-commit: HEAD~1..HEAD). `--all` 이면 매핑 전체,
   인자로 파일 경로를 직접 줄 수도 있다.
2. config.yaml 의 `pages` 매핑에 있는 파일만 대상.
3. 각 파일을 Notion blocks 로 변환 (상단 출처 callout + 본문 + GitHub 코드 카드).
4. 대상 페이지의 기존 children 을 모두 삭제하고 새 블록을 append (전량 교체 = 미러).

정본 규칙: 30-References/notion-mcp.md
- vault 가 단일 진실 공급원, Notion 은 미러.
- 토큰은 vault 밖 (~/.config/notion-sync/token, mode 600) 또는 환경변수 NOTION_TOKEN.

사용:
    python sync.py                 # HEAD~1..HEAD 변경분 동기
    python sync.py --all           # 매핑된 모든 페이지 동기
    python sync.py 10-Projects/pytorch-study.md   # 특정 파일만
    python sync.py --all --dry-run # API 호출 없이 변환 결과만 점검
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

from converter import markdown_to_blocks
from github_link import code_link_blocks

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.yaml")


def log(msg: str) -> None:
    print(f"[notion-sync] {msg}", flush=True)


# ── 설정·토큰 ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_token(cfg: dict) -> str:
    token = os.environ.get("NOTION_TOKEN")
    if token:
        return token.strip()
    path = os.path.expanduser(cfg.get("token_path", "~/.config/notion-sync/token"))
    if not os.path.isfile(path):
        sys.exit(
            f"[notion-sync] 토큰을 찾을 수 없습니다: 환경변수 NOTION_TOKEN 도 없고 {path} 도 없음.\n"
            "  notion-mcp.md '사용자 작업' 절차로 Integration Token 을 발급해 저장하세요."
        )
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ── 변경 파일 검출 ─────────────────────────────────────────────────────────

def changed_md_files(cfg: dict) -> list[str]:
    base = cfg.get("sync", {}).get("diff_base", "HEAD~1")
    head = cfg.get("sync", {}).get("diff_head", "HEAD")
    try:
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, "diff", "--name-only", f"{base}..{head}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        # 첫 커밋 등으로 base 가 없으면 빈 목록 (안전)
        log(f"git diff {base}..{head} 실패 — 변경 검출 건너뜀 (--all 로 전체 동기 가능).")
        return []
    return [ln for ln in out.splitlines() if ln.endswith(".md")]


def resolve_targets(args, cfg: dict) -> list[str]:
    pages = cfg.get("pages", {})
    if args.files:
        requested = [_normalize(f) for f in args.files]
    elif args.all:
        requested = list(pages.keys())
    else:
        requested = changed_md_files(cfg)

    targets, skipped = [], []
    for rel in requested:
        (targets if rel in pages else skipped).append(rel)
    for rel in skipped:
        log(f"매핑 없음 — 건너뜀: {rel}")
    return targets


def _normalize(path: str) -> str:
    """절대/상대 경로를 레포 루트 기준 상대경로로."""
    ap = os.path.abspath(path)
    if ap.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(ap, REPO_ROOT)
    return path


# ── 블록 빌드 ─────────────────────────────────────────────────────────────

def build_blocks(rel_path: str, cfg: dict) -> list[dict]:
    abs_path = os.path.join(REPO_ROOT, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        md_text = f.read()

    blocks: list[dict] = []
    if cfg.get("sync", {}).get("source_callout", True):
        blocks.append(_source_callout(rel_path))
    blocks.extend(markdown_to_blocks(md_text))
    blocks.extend(code_link_blocks(rel_path, md_text, REPO_ROOT, cfg["github"]))
    return blocks


def _source_callout(rel_path: str) -> dict:
    gh_hint = f"vault 정본: {rel_path} — Notion 은 미러 (직접 편집 금지, vault → git → Notion)"
    return {
        "object": "block", "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": "📌"},
            "color": "gray_background",
            "rich_text": [{"type": "text", "text": {"content": gh_hint}}],
        },
    }


# ── Notion 교체 ───────────────────────────────────────────────────────────

# 삭제하면 하위 페이지/DB 자체가 휴지통으로 가므로 절대 지우지 않고 보존한다.
PRESERVE_TYPES = {"child_page", "child_database"}


def replace_page_content(client, page_id: str, blocks: list[dict], batch_size: int) -> None:
    # 1) 기존 children 조회 → 콘텐츠 블록만 삭제 (child_page/child_database 는 보존)
    cursor = None
    existing = []
    while True:
        resp = client.blocks.children.list(page_id, start_cursor=cursor) if cursor \
            else client.blocks.children.list(page_id)
        existing.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    deleted = preserved = 0
    for blk in existing:
        if blk.get("type") in PRESERVE_TYPES:
            preserved += 1
            continue
        client.blocks.delete(blk["id"])
        deleted += 1
    log(f"  기존 블록 {deleted}개 삭제, 하위 페이지 {preserved}개 보존")

    # 2) 새 블록 append (배치)
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i:i + batch_size]
        client.blocks.children.append(page_id, children=chunk)
    log(f"  새 블록 {len(blocks)}개 추가")


# ── 엔트리포인트 ───────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="vault .md → Notion 미러 동기")
    ap.add_argument("files", nargs="*", help="동기할 .md 경로 (생략 시 git diff 기반)")
    ap.add_argument("--all", action="store_true", help="매핑된 모든 페이지 동기")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 변환만 점검")
    args = ap.parse_args()

    cfg = load_config()
    targets = resolve_targets(args, cfg)
    if not targets:
        log("동기 대상 없음.")
        return 0
    log(f"대상 {len(targets)}개: {', '.join(targets)}")

    batch_size = cfg.get("sync", {}).get("batch_size", 100)

    if args.dry_run:
        for rel in targets:
            blocks = build_blocks(rel, cfg)
            log(f"[dry-run] {rel} → 블록 {len(blocks)}개 (page_id={cfg['pages'][rel]})")
        return 0

    from notion_client import Client  # 실제 동기 시에만 import
    client = Client(auth=load_token(cfg))

    failed = 0
    for rel in targets:
        page_id = cfg["pages"][rel]
        log(f"동기: {rel} → {page_id}")
        try:
            blocks = build_blocks(rel, cfg)
            replace_page_content(client, page_id, blocks, batch_size)
        except Exception as e:  # noqa: BLE001 — 한 파일 실패가 나머지를 막지 않게
            failed += 1
            log(f"  ✗ 실패: {e}")
    log("완료." if not failed else f"완료 (실패 {failed}개).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
