"""핸드오프 계약 (HARNESS 골격 2: 컨텍스트 관리).

단계 사이에 비정형 dict를 흘리지 않고 타입화된 계약으로 넘긴다.
- PlanRequest: API 입력
- Intent: Haiku(경량 워커)가 분석한 의도 — Step 1 → Step 2/3로 흘러가는 핸드오프
- Activity / DayPlan / Plan: 최종 출력 구조 (검증층이 보정하는 대상)
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    """API 입력. 자연어 요청 + 인원/일수/예산."""
    query: str = Field(..., description="자연어 요청 예: '주말에 가족이랑 실내 위주로 저렴하게'")
    days: int = Field(1, ge=1, le=7, description="여행/나들이 일수")
    adults: int = Field(1, ge=1)
    children: int = Field(0, ge=0)
    budget: int | None = Field(None, description="1인 총예산(원). 선택.")


class Intent(BaseModel):
    """Haiku가 분석한 의도 (핸드오프 계약).

    - keywords: 검색/매칭용 키워드
    - include_categories / exclude_categories: 카탈로그 필터
    - indoor_only: 실내 선호 여부 (None=무관)
    - style_hint: 플랜 생성 프롬프트로 흘러가는 톤/스타일 힌트
    """
    keywords: list[str] = Field(default_factory=list)
    include_categories: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(default_factory=list)
    indoor_only: bool | None = None
    style_hint: str = ""


class Activity(BaseModel):
    itemId: int
    name: str = ""
    adultPrice: int = 0
    childPrice: int = 0


class DayPlan(BaseModel):
    day: int
    activities: list[Activity] = Field(default_factory=list)


class Plan(BaseModel):
    title: str = ""
    itinerary: list[DayPlan] = Field(default_factory=list)
    totalPrice: int = 0  # 항상 코드가 재계산 — LLM 값 신뢰 안 함


class PlanResponse(BaseModel):
    plan: Plan
    error: str | None = None  # 부분 실패 시 사용자 안내
