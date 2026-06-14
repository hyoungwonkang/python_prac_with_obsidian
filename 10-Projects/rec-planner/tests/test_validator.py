"""검증 루프 단위 테스트 (HARNESS 부록 A: 가장 먼저 만들 것).

anthropic·pydantic·fastapi 미설치여도 통과해야 한다 — validator/catalog만 의존.
실행: rec-planner 디렉터리에서  python -m pytest tests/test_validator.py -v
"""
import json

from core import validator as v
from core.catalog import valid_item_ids, catalog_index


# ---- parse_llm_response: 3전략 캐스케이드 ----

def test_parse_code_fence():
    text = '설명...\n```json\n{"a": 1}\n```\n뒷말'
    assert v.parse_llm_response(text) == {"a": 1}


def test_parse_direct():
    assert v.parse_llm_response('{"a": 1}') == {"a": 1}


def test_parse_embedded_brackets():
    text = '추천 결과입니다: {"itinerary": []} 이상입니다.'
    assert v.parse_llm_response(text) == {"itinerary": []}


def test_parse_failure_raises():
    try:
        v.parse_llm_response("JSON이 전혀 없음")
    except json.JSONDecodeError:
        return
    assert False, "깨진 입력에 JSONDecodeError를 raise해야 한다"


# ---- validate_plan: 환각 제거 + 가격 교정 + total 재계산 ----

def test_hallucinated_item_removed():
    ids, idx = valid_item_ids(), catalog_index()
    real = next(iter(ids))
    parsed = {"itinerary": [{"day": 1, "activities": [
        {"itemId": real},
        {"itemId": 999999},  # 환각 — 카탈로그에 없음
    ]}]}
    out = v.validate_plan(parsed, ids, idx)
    got = [a["itemId"] for a in out["itinerary"][0]["activities"]]
    assert got == [real], "환각 itemId는 제거되어야 한다"


def test_total_price_recalculated_ignoring_llm():
    ids, idx = valid_item_ids(), catalog_index()
    paid = next((i for i in ids if idx[i]["adultPrice"] > 0), None)
    assert paid is not None
    parsed = {"itinerary": [{"day": 1, "activities": [{"itemId": paid}]}],
              "totalPrice": 99999999}  # LLM 거짓 합계
    out = v.validate_plan(parsed, ids, idx, adults=2, children=0)
    assert out["totalPrice"] == idx[paid]["adultPrice"] * 2, "totalPrice는 코드가 재계산해야 한다"


def test_price_corrected_from_catalog():
    ids, idx = valid_item_ids(), catalog_index()
    paid = next((i for i in ids if idx[i]["adultPrice"] > 0), None)
    parsed = {"itinerary": [{"day": 1, "activities": [
        {"itemId": paid, "name": "조작된이름", "adultPrice": 1},  # LLM 거짓 가격
    ]}]}
    out = v.validate_plan(parsed, ids, idx)
    act = out["itinerary"][0]["activities"][0]
    assert act["adultPrice"] == idx[paid]["adultPrice"], "가격은 카탈로그 정본으로 교정"
    assert act["name"] == idx[paid]["name"]


def test_dedup_same_day():
    ids, idx = valid_item_ids(), catalog_index()
    real = next(iter(ids))
    parsed = {"itinerary": [{"day": 1, "activities": [{"itemId": real}, {"itemId": real}]}]}
    out = v.validate_plan(parsed, ids, idx)
    assert len(out["itinerary"][0]["activities"]) == 1, "같은 날 중복 itemId 제거"


def test_cap_per_day():
    ids, idx = valid_item_ids(), catalog_index()
    many = list(ids)[: v.MAX_ACTIVITIES_PER_DAY + 3]
    parsed = {"itinerary": [{"day": 1, "activities": [{"itemId": i} for i in many]}]}
    out = v.validate_plan(parsed, ids, idx)
    assert len(out["itinerary"][0]["activities"]) == v.MAX_ACTIVITIES_PER_DAY


# ---- check_plan_violations ----

def test_violation_wrong_day_count():
    plan = {"itinerary": [{"day": 1, "activities": [{"itemId": 1}]}], "totalPrice": 0}
    assert v.check_plan_violations(plan, days=2)  # 2일 요청에 1일 → 위반


def test_violation_budget_exceeded():
    plan = {"itinerary": [{"day": 1, "activities": [{"itemId": 1}]}], "totalPrice": 50000}
    assert v.check_plan_violations(plan, days=1, budget=10000)


def test_no_violation_when_clean():
    ids, idx = valid_item_ids(), catalog_index()
    free = next((i for i in ids if idx[i]["adultPrice"] == 0), None)
    plan = {"itinerary": [{"day": 1, "activities": [{"itemId": free}]}], "totalPrice": 0}
    assert v.check_plan_violations(plan, days=1, budget=0) == []
