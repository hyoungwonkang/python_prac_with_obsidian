"""순수 추천 파이프라인 (HARNESS 골격 1: 제어 흐름 + 골격 5: 검증 루프).

웹 경계(FastAPI)와 분리된 순수 로직 — validator/catalog/guardrail에만 의존.
LLM 워커는 **주입(inject)** 받는다 → anthropic 없이 stub으로 전 단계 테스트 가능.

Step 0  입력 검열 (guardrail)        ← 위반 시 GuardrailBlockedError (→400)
Step 1  의도 분석 (주입된 워커)
Step 2  후보 필터 (로컬 카탈로그)
Step 3  플랜 생성 + verify→fix 루프  ← max_attempts 상한, 위반→교정 재호출
Step 4  최종 폴백 (결정적)
"""
from __future__ import annotations

import logging

from . import catalog, guardrail, validator

logger = logging.getLogger("rec-planner")

MAX_ATTEMPTS = 3  # verify→fix 루프 상한 (무한루프·비용폭주 방지)


def run_pipeline(query: str, days: int, adults: int, children: int,
                 budget: int | None = None, *,
                 analyze_intent, build_prompt, generate_plan) -> dict:
    """추천 플랜 dict를 반환. {"plan": {...}, "error": str|None}.

    analyze_intent(query) -> intent
    build_prompt(query, intent, candidates, days, adults, children) -> str
    generate_plan(user_message, attempt) -> str(JSON 추정)
    """
    # Step 0: 입력 검열 (예외는 호출부에서 400으로 매핑)
    guardrail.screen_input(query)

    valid_ids = catalog.valid_item_ids()
    item_idx = catalog.catalog_index()

    # Step 1: 의도 분석 (경량 워커)
    intent = analyze_intent(query)
    logger.info("Step1 intent ok")

    # Step 2: 후보 필터
    candidates = catalog.filter_candidates(intent)
    logger.info("Step2 candidates: %d개", len(candidates))

    base_prompt = build_prompt(query, intent, candidates, days, adults, children)

    # Step 3: 생성 → 파싱 → 검증 → 위반 점검 → (위반 시) 교정 재호출
    last_validated: dict | None = None
    user_message = base_prompt
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = generate_plan(user_message, attempt=attempt)
            parsed = validator.parse_llm_response(raw)          # 3전략 파싱
        except Exception as e:                                  # JSON 깨짐/호출 실패 → 교정 재시도
            logger.warning("attempt %d 실패: %s", attempt, e)
            user_message = base_prompt + "\n\n[교정] 이전 출력은 JSON이 아니었다. 순수 JSON만 출력하라."
            continue

        validated = validator.validate_plan(                    # 거부+교정
            parsed, valid_ids, item_idx, adults, children
        )
        last_validated = validated

        violations = validator.check_plan_violations(validated, days, budget)
        if not violations:                                      # 통과
            logger.info("attempt %d 통과", attempt)
            return {"plan": validated, "error": None}

        logger.info("attempt %d 위반 %d건 → 재시도", attempt, len(violations))
        user_message = base_prompt + "\n\n[교정 지시]\n- " + "\n- ".join(violations)

    # Step 4: 상한 도달 → 결정적 폴백 (안전한 최소 유효 결과)
    logger.warning("상한 도달 → 폴백")
    fallback = last_validated or deterministic_plan(candidates, days, adults, children, item_idx)
    return {"plan": fallback, "error": "LLM이 제약을 모두 만족하지 못해 보정된 결과입니다."}


def deterministic_plan(candidates: list[dict], days: int, adults: int, children: int,
                       item_idx: dict[int, dict]) -> dict:
    """LLM 전면 실패 시: 후보를 일자별로 균등 배분한 안전한 플랜 (LLM 0회)."""
    valid_ids = set(item_idx.keys())
    per_day = min(validator.MAX_ACTIVITIES_PER_DAY, 2)
    pool = candidates[: days * per_day] or candidates
    itinerary, cursor = [], 0
    for day in range(1, days + 1):
        chunk = pool[cursor:cursor + per_day]
        cursor += per_day
        itinerary.append({"day": day, "activities": [{"itemId": c["id"]} for c in chunk]})
    return validator.validate_plan(
        {"title": "기본 추천 플랜", "itinerary": itinerary},
        valid_ids, item_idx, adults, children,
    )
