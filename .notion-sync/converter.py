"""converter.py — Markdown(mistune AST) → Notion blocks 변환.

vault `.md` 한 파일을 Notion API가 받는 block 객체 리스트로 바꾼다.
정본 규칙: 30-References/notion-mcp.md

지원 블록: heading_1~3, paragraph, bulleted/numbered list, to_do,
code, quote, divider, table. 인라인: bold/italic/code/strikethrough/link.

Notion 제약:
- rich_text 한 객체의 text.content 는 2000자 이하 → 길면 분할.
- heading 은 h3 까지 → h4+ 는 굵은 paragraph 로 강등.
- 한 블록의 children(중첩 리스트)은 재귀로 처리.
"""

from __future__ import annotations

import mistune

# Notion rich_text content 한 객체 최대 길이
MAX_TEXT_LEN = 2000


def markdown_to_blocks(md_text: str) -> list[dict]:
    """Markdown 문자열 → Notion block dict 리스트."""
    parse = mistune.create_markdown(
        renderer=None, plugins=["table", "strikethrough", "task_lists"]
    )
    tokens = parse(md_text)
    blocks: list[dict] = []
    for tok in tokens:
        blocks.extend(_token_to_blocks(tok))
    return blocks


# ── 블록 토큰 → Notion 블록 ──────────────────────────────────────────────

def _token_to_blocks(tok: dict) -> list[dict]:
    t = tok.get("type")

    if t == "heading":
        level = tok.get("attrs", {}).get("level", 1)
        rich = _inline(tok.get("children", []))
        if level <= 3:
            key = f"heading_{level}"
            return [{"object": "block", "type": key, key: {"rich_text": rich}}]
        # h4+ → 굵은 paragraph
        for r in rich:
            r["annotations"]["bold"] = True
        return [_paragraph(rich)]

    if t in ("paragraph", "block_text"):
        return [_paragraph(_inline(tok.get("children", [])))]

    if t == "block_code":
        info = (tok.get("attrs", {}).get("info") or "").strip()
        lang = _notion_lang(info.split()[0] if info else "")
        raw = tok.get("raw", "")
        if raw.endswith("\n"):
            raw = raw[:-1]
        return [{
            "object": "block", "type": "code",
            "code": {"language": lang, "rich_text": _split_text(raw)},
        }]

    if t == "block_quote":
        rich = []
        children_blocks = []
        for child in tok.get("children", []):
            if child.get("type") in ("paragraph", "block_text"):
                if rich:
                    rich.append({"type": "text", "text": {"content": "\n"}})
                rich.extend(_inline(child.get("children", [])))
            else:
                children_blocks.extend(_token_to_blocks(child))
        quote = {"object": "block", "type": "quote", "quote": {"rich_text": rich}}
        if children_blocks:
            quote["quote"]["children"] = children_blocks
        return [quote]

    if t == "thematic_break":
        return [{"object": "block", "type": "divider", "divider": {}}]

    if t == "list":
        ordered = tok.get("attrs", {}).get("ordered", False)
        out = []
        for item in tok.get("children", []):
            out.extend(_list_item(item, ordered))
        return out

    if t == "blank_line":
        return []

    if t == "table":
        tbl = _table(tok)
        return [tbl] if tbl else []

    # 알 수 없는 토큰: children 있으면 펼치고, 아니면 무시
    out = []
    for child in tok.get("children", []) or []:
        out.extend(_token_to_blocks(child))
    return out


def _list_item(item: dict, ordered: bool) -> list[dict]:
    """list_item 토큰 → bulleted/numbered/to_do 블록 (중첩 children 포함)."""
    rich: list[dict] = []
    children_blocks: list[dict] = []
    for child in item.get("children", []):
        ct = child.get("type")
        if ct in ("paragraph", "block_text"):
            if rich:
                rich.append({"type": "text", "text": {"content": "\n"}})
            rich.extend(_inline(child.get("children", [])))
        elif ct == "list":
            children_blocks.extend(_token_to_blocks(child))
        else:
            children_blocks.extend(_token_to_blocks(child))

    # GFM task list: "- [ ] " / "- [x] " → to_do
    checked = _task_state(item, rich)
    if checked is not None:
        key, payload = "to_do", {"rich_text": rich, "checked": checked}
    elif ordered:
        key, payload = "numbered_list_item", {"rich_text": rich}
    else:
        key, payload = "bulleted_list_item", {"rich_text": rich}

    if children_blocks:
        payload["children"] = children_blocks
    return [{"object": "block", "type": key, key: payload}]


def _task_state(item: dict, rich: list[dict]):
    """task list 여부 판정. mistune task_lists 미사용 시 텍스트 앞부분으로 폴백."""
    attrs = item.get("attrs", {})
    if "checked" in attrs:
        return bool(attrs["checked"])
    if not rich:
        return None
    head = rich[0]["text"]["content"]
    if head.startswith("[ ] "):
        rich[0]["text"]["content"] = head[4:]
        return False
    if head.startswith("[x] ") or head.startswith("[X] "):
        rich[0]["text"]["content"] = head[4:]
        return True
    return None


def _table(tok: dict) -> dict | None:
    """table 토큰 → Notion table 블록."""
    head = None
    body_rows: list[dict] = []
    for section in tok.get("children", []):
        st = section.get("type")
        if st == "table_head":
            head = section
        elif st == "table_body":
            body_rows = section.get("children", [])

    def cells(row_tok) -> list[list[dict]]:
        return [_inline(c.get("children", [])) for c in row_tok.get("children", [])]

    rows = []
    if head is not None:
        rows.append(cells(head))
    rows.extend(cells(r) for r in body_rows)
    if not rows:
        return None

    width = max(len(r) for r in rows)
    table_rows = []
    for r in rows:
        padded = r + [[] for _ in range(width - len(r))]
        table_rows.append({
            "object": "block", "type": "table_row",
            "table_row": {"cells": padded},
        })
    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": head is not None,
            "has_row_header": False,
            "children": table_rows,
        },
    }


# ── 인라인 토큰 → rich_text ───────────────────────────────────────────────

def _inline(tokens: list[dict], ann: dict | None = None, link: str | None = None) -> list[dict]:
    ann = ann or _ann()
    out: list[dict] = []
    for tok in tokens:
        t = tok.get("type")
        if t == "text":
            out.extend(_text(tok.get("raw", ""), ann, link))
        elif t == "codespan":
            a = dict(ann); a["code"] = True
            out.extend(_text(tok.get("raw", ""), a, link))
        elif t == "strong":
            a = dict(ann); a["bold"] = True
            out.extend(_inline(tok.get("children", []), a, link))
        elif t == "emphasis":
            a = dict(ann); a["italic"] = True
            out.extend(_inline(tok.get("children", []), a, link))
        elif t == "strikethrough":
            a = dict(ann); a["strikethrough"] = True
            out.extend(_inline(tok.get("children", []), a, link))
        elif t == "link":
            href = tok.get("attrs", {}).get("url", link)
            out.extend(_inline(tok.get("children", []), ann, href))
        elif t == "image":
            # 이미지 인라인은 alt 텍스트 + URL 링크로 표현
            alt = tok.get("children", [])
            url = tok.get("attrs", {}).get("url")
            out.extend(_inline(alt, ann, url) if alt else _text(url or "", ann, url))
        elif t in ("linebreak", "softbreak"):
            out.extend(_text("\n", ann, link))
        elif t == "inline_html":
            out.extend(_text(tok.get("raw", ""), ann, link))
        else:
            if tok.get("children"):
                out.extend(_inline(tok["children"], ann, link))
            elif tok.get("raw"):
                out.extend(_text(tok["raw"], ann, link))
    return out


def _text(content: str, ann: dict, link: str | None) -> list[dict]:
    """content 를 2000자 단위로 잘라 rich_text 객체 리스트로."""
    if content == "":
        return []
    out = []
    for i in range(0, len(content), MAX_TEXT_LEN):
        chunk = content[i:i + MAX_TEXT_LEN]
        text_obj = {"content": chunk}
        if link:
            text_obj["link"] = {"url": link}
        out.append({"type": "text", "text": text_obj, "annotations": dict(ann)})
    return out


def _split_text(content: str) -> list[dict]:
    """code 블록 등 순수 텍스트를 2000자 단위 rich_text 로."""
    return _text(content, _ann(), None) or [{"type": "text", "text": {"content": ""}}]


def _paragraph(rich: list[dict]) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich}}


def _ann() -> dict:
    return {
        "bold": False, "italic": False, "strikethrough": False,
        "underline": False, "code": False, "color": "default",
    }


# Notion code 블록이 지원하는 언어로 매핑 (미지원은 plain text)
_NOTION_LANGS = {
    "python", "javascript", "typescript", "bash", "shell", "json", "yaml",
    "markdown", "sql", "html", "css", "java", "c", "c++", "go", "rust",
    "ruby", "php", "diff", "docker", "graphql", "toml", "plain text",
}
_LANG_ALIAS = {
    "py": "python", "js": "javascript", "ts": "typescript", "sh": "bash",
    "zsh": "bash", "console": "shell", "yml": "yaml", "md": "markdown",
    "dockerfile": "docker", "text": "plain text", "txt": "plain text",
    "": "plain text",
}


def _notion_lang(info: str) -> str:
    info = (info or "").lower()
    info = _LANG_ALIAS.get(info, info)
    return info if info in _NOTION_LANGS else "plain text"
