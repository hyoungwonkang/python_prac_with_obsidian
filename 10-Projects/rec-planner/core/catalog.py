"""로컬 카탈로그 로더 (travel-rag의 ES/SKU DB 대체).

검증 루프의 '정답지' — LLM이 추천한 itemId가 실제로 존재하는지 대조하는 기준.
외부 의존성 0 (json·pathlib만) → 검증 루프 테스트가 anthropic 없이도 돈다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """catalog.json 전체를 로드 (캐시)."""
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def catalog_index() -> dict[int, dict]:
    """id → item dict 인덱스. 환각 검증·가격 교정에 사용."""
    return {item["id"]: item for item in load_catalog()}


def valid_item_ids() -> set[int]:
    """존재하는 itemId 집합 (환각 검증 기준)."""
    return set(catalog_index().keys())


def filter_candidates(intent) -> list[dict]:
    """Intent로 후보를 좁힌다 (Step 2: 검색 대체).

    카테고리 포함/제외 + 실내 선호만 반영하는 단순 휴리스틱.
    travel-rag의 ES 하이브리드 검색을 로컬 필터로 대체한 것.
    """
    items = load_catalog()
    inc = {c for c in getattr(intent, "include_categories", [])}
    exc = {c for c in getattr(intent, "exclude_categories", [])}
    indoor_only = getattr(intent, "indoor_only", None)

    def keep(item: dict) -> bool:
        if exc and item["category"] in exc:
            return False
        if inc and item["category"] not in inc:
            return False
        if indoor_only is True and not item["indoor"]:
            return False
        if indoor_only is False and item["indoor"]:
            return False
        return True

    filtered = [i for i in items if keep(i)]
    # 필터가 전부 걸러내면 전체를 후보로 (빈 결과보다 낫다 — graceful degradation)
    return filtered or items
