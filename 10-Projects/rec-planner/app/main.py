"""FastAPI 앱 진입점 (HARNESS 골격 6: 장애 처리 — API 경계 매핑).

실행: rec-planner 디렉터리에서
    uvicorn app.main:app --reload
    → http://localhost:8000/docs 에서 POST /plan 시도
"""
from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import plan

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="rec-planner", description="검증된 추천 플래너 (HARNESS 경량 적용)")

# CORS — 프론트 분리 대비 (todo-app 패턴)
_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plan.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# 비정형 예외 → 사용자 안내 메시지 + 적절한 상태코드 (조용히 500 던지지 않음)
@app.exception_handler(json.JSONDecodeError)
async def json_error_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(status_code=422,
                        content={"error": "LLM 응답을 해석하지 못했습니다. 다시 시도해 주세요."})
