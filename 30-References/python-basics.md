# Python 기초 개념 정리

todo-app 프로젝트를 진행하며 만나는 기본 개념들.

## 문법 기초

### `class`
- **클래스(설계도)를 정의**하는 키워드.
- 클래스로부터 **인스턴스(객체)**를 만들어 사용.
- `class Todo(Base):` → Base를 **상속**한 Todo 클래스 정의.
- 상속: 부모 클래스의 기능을 물려받아 확장하는 것.

### `def` / `async def`
- `def` — **함수를 정의**하는 키워드. define의 약자.
- `async def` — **비동기 함수** 정의. DB 접속 같은 대기 작업을 효율적으로 처리.
- 예: `def hello(name):` → 일반 함수, `async def get_todos():` → 비동기 함수

### `return`
- 함수의 결과를 돌려줌. `return`이 없으면 `None` 반환.
- 예: `return {"status": "ok"}` → 딕셔너리를 결과로 돌려줌.

### `import` / `from ... import ...`
- 다른 모듈(파일)의 코드를 가져옴.
- `import os` → os 모듈 전체를 가져옴 (`os.getenv(...)`)
- `from os import getenv` → getenv만 가져옴 (`getenv(...)` 바로 사용)
- `from app.models import Base` → 계층적 import (패키지.모듈.이름)

### `await`
- 비동기 함수(`async def`) 안에서 대기 작업의 완료를 기다림.
- 예: `await session.commit()` → DB 저장이 끝날 때까지 기다림.
- `async def` 안에서만 사용 가능.

### 데코레이터 (`@`)
- 함수나 클래스에 기능을 덧붙이는 문법. 함수 위에 `@이름`으로 표시.
- 예: `@app.get("/health")` → 이 함수를 GET /health 요청에 연결.
- FastAPI에서 라우팅(URL ↔ 함수 연결)에 핵심적으로 사용.

## 표준 라이브러리

### `getenv(name, default)`
- `from os import getenv`
- 환경변수를 읽어옴. 없으면 default 반환.
- 예: `getenv("DATABASE_URL", "sqlite:///./todo.db")` → 환경변수가 없으면 SQLite 사용

### `datetime`
- `from datetime import datetime`
- 날짜/시간을 다루는 타입. `2026-04-26T19:25:17` 같은 값을 담음.
- 예: `datetime.now()` → 현재 시각

## 외부 라이브러리

### `BaseModel` (Pydantic)
- `from pydantic import BaseModel`
- 이걸 상속한 클래스는 자동으로 타입 검증 + JSON 변환이 됨.
- FastAPI가 요청/응답 데이터를 자동 검증하는 핵심 기반.

### Alembic
- DB 스키마 마이그레이션 도구. SQLAlchemy와 함께 사용.
- 모델(Python 클래스)이 바뀌면, Alembic이 그 차이를 감지해서 DB 테이블을 자동으로 맞춰줌.
- `alembic revision --autogenerate` → 변경 감지해서 마이그레이션 파일 생성
- `alembic upgrade head` → 마이그레이션 적용 (테이블 생성/수정)
- SQLite → Postgres 전환 시에도 동일한 마이그레이션 파일로 양쪽 적용 가능.

### asyncpg
- PostgreSQL 전용 **비동기 드라이버**. `async def` + `await`로 DB에 접속.
- SQLAlchemy URL에서 `postgresql+asyncpg://...`로 지정하면 자동으로 사용됨.
- `aiosqlite`가 SQLite용 비동기 드라이버인 것처럼, `asyncpg`는 Postgres용.
- DB 대기 중에도 다른 요청을 처리할 수 있어 FastAPI의 비동기 구조와 맞음.

### `load_dotenv()` (python-dotenv)
- `from dotenv import load_dotenv`
- `.env` 파일의 내용을 환경변수로 로드. `getenv()`로 읽을 수 있게 해줌.
- 비밀번호·API키 등을 코드에 직접 쓰지 않기 위해 사용.

## Python 구조

### `__init__.py`
- 폴더를 **Python 패키지**로 만드는 표식 파일. 내용은 보통 비어있음.
- 이 파일이 있어야 `from app.main import app` 같은 계층적 import 가능.

### 가상환경 (venv / uv venv)
- 프로젝트마다 독립된 패키지 공간을 만듦. 프로젝트 간 버전 충돌 방지.
- `.venv/` 폴더에 생성되고, `source .venv/bin/activate`로 활성화.

## Jinja2 템플릿 문법

HTML 안에서 Python 로직을 사용하기 위한 문법. FastAPI의 Jinja2Templates가 처리.

### `{% %}` — 제어문 (로직)
- `{% extends "base.html" %}` → 베이스 템플릿을 상속
- `{% block content %}...{% endblock %}` → 블록 정의/오버라이드
- `{% if todos %}...{% else %}...{% endif %}` → 조건문
- `{% for todo in todos %}...{% endfor %}` → 반복문

### `{{ }}` — 변수 출력
- `{{ todo.title }}` → todo 객체의 title 값을 HTML에 출력
- `{{ '✓' if todo.done else '○' }}` → 삼항 표현식

### `{# #}` — 주석
- `{# 이 부분은 HTML에 출력되지 않음 #}`

## 파일 확장자

### `.ini`
- **INI**tialisation의 약자. 설정 파일 형식.
- `[섹션]` 아래에 `키 = 값` 형태로 설정을 나열.
- `alembic.ini`가 대표적 예. DB URL, 로깅 레벨 등 Alembic 설정을 담음.
- Python 표준 라이브러리 `configparser`로 읽을 수 있음.

## 타입 힌트

### `str | None = None`
- "문자열 또는 None, 기본값은 None" 이라는 뜻.
- `description: str | None = None` → 선택 필드 (안 넘기면 None)
- `title: str` → 필수 필드

## 프로젝트 파일

### `requirements.txt`
- 프로젝트에 필요한 **외부 패키지 목록**을 적어놓은 파일.
- `pip install -r requirements.txt` → 목록의 패키지를 한 번에 설치.
- 다른 사람(또는 다른 PC)에서 프로젝트를 클론한 뒤, 같은 환경을 재현하기 위해 사용.
- 버전 고정 예: `fastapi==0.115.0`, 최소 버전 예: `fastapi>=0.115.0`
- `pip freeze > requirements.txt` → 현재 설치된 패키지를 파일로 내보내기.
