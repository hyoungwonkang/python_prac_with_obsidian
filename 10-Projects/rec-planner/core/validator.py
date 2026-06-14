"""검증 루프 본체 (HARNESS 골격 5: 하네스의 심장).

travel-rag `recommend/validator.py`의 경량 이식.
"엔진의 출력을 절대 신뢰하지 않는다. 검증층이 보정한다."

- parse_llm_response: 깨진 JSON 3전략 복구 (장애 처리, 골격 6)
- validate_plan: 환각 itemId 제거 + 가격 교정 + totalPrice 재계산
- check_plan_violations: 제약 위반 탐지 → verify→fix 루프의 fixup 재료

외부 의존성 0 (json·re) → anthropic 없이 테스트 가능.
"""
from __future__ import annotations

import json
import re

MAX_ACTIVITIES_PER_DAY = 4  # 출력 폭주 방지 cap


def parse_llm_response(response_text: str):
    """LLM 응답 텍스트 → JSON (dict/list). 3전략 캐스케이드.

    1) ```json ... ``` 마크다운 코드펜스
    2) 직접 파싱
    3) 첫 { 또는 [ 부터 괄호 중첩 카운팅으로 추출
    모두 실패하면 JSONDecodeError를 raise (조용히 삼키지 않음).
    """
    # 1) 코드펜스
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2) 직접
    stripped = response_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3) 괄호 중첩 카운팅
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(stripped)):
            if stripped[i] == start_char:
                depth += 1
            elif stripped[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON found in LLM response", response_text, 0)


def _coerce_plan(parsed) -> dict:
    """LLM이 낼 수 있는 여러 형태를 {itinerary:[...]} 로 정규화."""
    if isinstance(parsed, list):
        # day_plan 리스트 그대로 온 경우
        return {"itinerary": parsed}
    if isinstance(parsed, dict):
        if "itinerary" in parsed:
            return parsed
        for key in ("plan", "days", "result"):
            if key in parsed and isinstance(parsed[key], list):
                return {"itinerary": parsed[key], **{k: v for k, v in parsed.items() if k != key}}
    return {"itinerary": []}


def validate_plan(parsed, valid_ids: set[int], item_idx: dict[int, dict],
                  adults: int = 1, children: int = 0,
                  max_per_day: int = MAX_ACTIVITIES_PER_DAY) -> dict:
    """플랜을 검증·교정한 dict로 반환.

    - 환각 itemId(카탈로그에 없는) 활동 제거
    - name/가격을 카탈로그 정본으로 교정 (LLM 값 신뢰 안 함)
    - 같은 날 중복 itemId 제거
    - 하루 활동 수 cap 초과분 절단
    - totalPrice 항상 재계산 (산술 오류 방지)
    """
    plan = _coerce_plan(parsed)
    cleaned_itinerary: list[dict] = []

    for day_plan in plan.get("itinerary", []):
        if not isinstance(day_plan, dict):
            continue
        day = day_plan.get("day", len(cleaned_itinerary) + 1)
        seen: set[int] = set()
        valid_activities: list[dict] = []
        for act in day_plan.get("activities", []) or []:
            if not isinstance(act, dict):
                continue
            item_id = act.get("itemId")
            if item_id not in valid_ids:          # 환각 제거
                continue
            if item_id in seen:                    # 같은 날 중복 제거
                continue
            seen.add(item_id)
            src = item_idx[item_id]                # 카탈로그 정본으로 교정
            valid_activities.append({
                "itemId": item_id,
                "name": src["name"],
                "adultPrice": src.get("adultPrice", 0),
                "childPrice": src.get("childPrice", 0),
            })
            if len(valid_activities) >= max_per_day:  # cap
                break
        cleaned_itinerary.append({"day": day, "activities": valid_activities})

    # totalPrice 항상 재계산 (LLM 산술 오류 방지)
    total = sum(
        act["adultPrice"] * adults + act["childPrice"] * children
        for d in cleaned_itinerary
        for act in d["activities"]
    )

    return {
        "title": plan.get("title", "") if isinstance(plan.get("title"), str) else "",
        "itinerary": cleaned_itinerary,
        "totalPrice": total,
    }


def check_plan_violations(plan: dict, days: int, budget: int | None = None) -> list[str]:
    """검증 후에도 남는 제약 위반을 문자열 리스트로 반환.

    비어 있으면 통과. 비어 있지 않으면 verify→fix 루프가 이걸 교정 지시로 만든다.
    """
    violations: list[str] = []
    itinerary = plan.get("itinerary", [])

    if len(itinerary) != days:
        violations.append(f"itinerary는 정확히 {days}일이어야 하는데 {len(itinerary)}일이 왔다.")

    for d in itinerary:
        if not d.get("activities"):
            violations.append(f"{d.get('day')}일차에 활동이 0개다. 최소 1개 채워라.")

    if budget is not None and plan.get("totalPrice", 0) > budget:
        violations.append(f"totalPrice {plan['totalPrice']}원이 1인 예산 {budget}원을 초과했다. 더 저렴한 활동으로 교체하라.")

    return violations
