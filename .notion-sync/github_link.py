"""github_link.py — vault 노트가 참조하는 실습 코드 → GitHub 링크 카드 블록.

노트 본문에서 코드 파일명(백틱·위키링크)을 검출하고, 레포 안에 실제로
존재하는 파일만 골라 blob URL 기반 Notion bookmark 블록으로 만든다.
repo 가 public 이라 bookmark 카드가 항상 최신 코드 미리보기를 보여 준다.

정본 규칙: 30-References/notion-mcp.md (실습 코드 표시 = GitHub 임베드/링크)
"""

from __future__ import annotations

import os
import re

# 링크 카드로 만들 코드 확장자
CODE_EXT = (".py", ".ipynb", ".js", ".ts", ".sh", ".sql", ".yaml", ".yml")

# 백틱 인라인 코드 또는 위키링크 안에서 파일명 후보 추출
_FILE_RE = re.compile(r"`([^`\n]+?)`|\[\[([^\]\n]+?)\]\]")


def code_link_blocks(md_path: str, md_text: str, repo_root: str, gh: dict) -> list[dict]:
    """노트가 참조하는 실습 코드의 GitHub bookmark 블록 리스트.

    md_path: 레포 루트 기준 .md 경로 (예: "10-Projects/pytorch-study.md")
    repo_root: 레포 루트 절대경로
    gh: config 의 github 섹션 (owner/repo/branch)
    """
    index = _code_index(md_path, repo_root)
    if not index:
        return []

    referenced: list[str] = []
    seen: set[str] = set()
    for raw in _candidates(md_text):
        base = os.path.basename(raw.strip())
        relpath = index.get(base)
        if relpath and relpath not in seen:
            seen.add(relpath)
            referenced.append(relpath)

    if not referenced:
        return []

    referenced.sort()
    blocks: list[dict] = [{
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "실습 코드 (GitHub)"}}]},
    }]
    base_url = f"https://github.com/{gh['owner']}/{gh['repo']}/blob/{gh['branch']}"
    for relpath in referenced:
        url = f"{base_url}/{relpath}"
        blocks.append({
            "object": "block", "type": "bookmark",
            "bookmark": {
                "url": url,
                "caption": [{"type": "text", "text": {"content": relpath}}],
            },
        })
    return blocks


def _candidates(md_text: str):
    for m in _FILE_RE.finditer(md_text):
        token = m.group(1) or m.group(2) or ""
        # 위키링크의 alias( `|` ) 제거
        token = token.split("|", 1)[0]
        if token.lower().endswith(CODE_EXT):
            yield token


def _code_index(md_path: str, repo_root: str) -> dict[str, str]:
    """노트와 관련된 코드 디렉터리를 walk → {basename: 레포기준 상대경로}.

    검색 범위: 노트의 부모 폴더 + (노트명에서 .md 뗀) 동명 프로젝트 폴더.
    예: 10-Projects/pytorch-study.md → 10-Projects/ 와 10-Projects/pytorch-study/.
    """
    search_dirs = set()
    parent = os.path.dirname(md_path)
    if parent:
        search_dirs.add(parent)
    project_dir = md_path[:-3] if md_path.endswith(".md") else md_path
    search_dirs.add(project_dir)

    index: dict[str, str] = {}
    for rel_dir in search_dirs:
        abs_dir = os.path.join(repo_root, rel_dir)
        if not os.path.isdir(abs_dir):
            continue
        for root, dirs, files in os.walk(abs_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for fn in files:
                if fn.lower().endswith(CODE_EXT):
                    rel = os.path.relpath(os.path.join(root, fn), repo_root)
                    # 동명 파일이 여러 곳이면 경로 짧은 쪽(상위) 우선
                    if fn not in index or len(rel) < len(index[fn]):
                        index[fn] = rel
    return index
