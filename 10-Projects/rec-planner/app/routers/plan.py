"""추천 파이프라인 (HARNESS 골격 1: 제어 흐름 + 골격 5: 검증 루프).

Step 1  의도 분석 (Haiku)           ← 핸드오프 Intent 생성
Step 2  후보 필터 (로컬 카탈로그)     ← ES 검색 대체
Step 3  플랜 생성 + verify→fix 루프  ← max_attempts 상한, 위반→교정 재호출
Step 4  최종 폴백 (결정적)           ← 상한 도달 시 안전한 결과 보장
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from core import catalog, llm, validator
from core.schema import Plan, PlanRequest, PlanResponse

logger = logging.getLogger("rec-planner")
router = APIRouter()

MAX_ATTEMPTS = 3  # verify→fix 루프 상한 (무한루프·비용폭주 방지)


@router.post("/plan", response_model=PlanResponse)
def make_plan(req: PlanRequest) -> PlanResponse:
    valid_ids = catalog.valid_item_ids()
    item_idx = catalog.catalog_index()

    # Step 1: 의도 분석 (경량 워커, 실패 시 휴리스틱 강등)
    intent = llm.analyze_intent(req.query)
    logger.info("Step1 intent: %s", intent.model_dump())

    # Step 2: 후보 필터
    candidates = catalog.filter_candidates(intent)
    logger.info("Step2 candidates: %d개", len(candidates))

    base_prompt = llm.build_plan_prompt(
        req.query, intent, candidates, req.days, req.adults, req.children
    )

    # Step 3: 생성 → 파싱 → 검증 → 위반 점검 → (위반 시) 교정 재호출
    last_validated: dict | None = None
    user_message = base_prompt
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = llm.generate_plan(user_message, attempt=attempt)
            parsed = validator.parse_llm_response(raw)          # 3전략 파싱
        except Exception as e:                                  # JSON 깨짐 → 교정 재시도
            logger.warning("attempt %d 파싱 실패: %s", attempt, e)
            user_message = base_prompt + "\n\n[교정] 이전 출력은 JSON이 아니었다. 순수 JSON만 출력하라."
            continue

        validated = validator.validate_plan(                    # 거부+교정
            parsed, valid_ids, item_idx, req.adults, req.children
        )
        last_validated = validated

        violations = validator.check_plan_violations(validated, req.days, req.budget)
        if not violations:                                      # 통과
            logger.info("attempt %d 통과", attempt)
            return PlanResponse(plan=Plan(**validated))

        # 위반을 교정 지시로 만들어 다음 attempt에 주입
        logger.info("attempt %d 위반 %d건 → 재시도", attempt, len(violations))
        user_message = base_prompt + "\n\n[교정 지시]\n- " + "\n- ".join(violations)

    # Step 4: 상한 도달 → 결정적 폴백 (안전한 최소 유효 결과)
    logger.warning("상한 도달 → 폴백")
    fallback = last_validated or _deterministic_plan(candidates, req, item_idx)
    return PlanResponse(
        plan=Plan(**fallback),
        error="LLM이 제약을 모두 만족하지 못해 보정된 결과입니다.",
    )


def _deterministic_plan(candidates: list[dict], req: PlanRequest,
                        item_idx: dict[int, dict]) -> dict:
    """LLM 전면 실패 시: 후보를 일자별로 균등 배분한 안전한 플랜 (LLM 0회)."""
    valid_ids = set(item_idx.keys())
    per_day = min(validator.MAX_ACTIVITIES_PER_DAY, 2)
    pool = candidates[: req.days * per_day] or candidates
    itinerary = []
    cursor = 0
    for day in range(1, req.days + 1):
        chunk = pool[cursor:cursor + per_day]
        cursor += per_day
        itinerary.append({"day": day, "activities": [{"itemId": c["id"]} for c in chunk]})
    # validate_plan으로 가격/total을 정본대로 채운다
    return validator.validate_plan(
        {"title": "기본 추천 플랜", "itinerary": itinerary},
        valid_ids, item_idx, req.adults, req.children,
    )
