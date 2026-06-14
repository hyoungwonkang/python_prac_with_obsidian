"""추천 라우터 — 순수 파이프라인(core/pipeline.py)을 감싸는 얇은 FastAPI 어댑터.

제어 흐름·검증 루프 본체는 core/pipeline.py에 있다 (웹 경계와 분리).
여기서는 PlanRequest → run_pipeline → PlanResponse 변환만 담당.
GuardrailBlockedError는 app/main.py가 400으로 매핑한다 (골격 6).
"""
from __future__ import annotations

from fastapi import APIRouter

from core import llm, pipeline
from core.schema import Plan, PlanRequest, PlanResponse

router = APIRouter()


@router.post("/plan", response_model=PlanResponse)
def make_plan(req: PlanRequest) -> PlanResponse:
    result = pipeline.run_pipeline(
        req.query, req.days, req.adults, req.children, req.budget,
        analyze_intent=llm.analyze_intent,
        build_prompt=llm.build_plan_prompt,
        generate_plan=llm.generate_plan,
    )
    return PlanResponse(plan=Plan(**result["plan"]), error=result["error"])
