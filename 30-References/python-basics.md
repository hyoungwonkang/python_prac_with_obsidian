# Python 기초 개념 정리

todo-app 프로젝트를 진행하며 만나는 기본 개념들.

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

## 타입 힌트

### `str | None = None`
- "문자열 또는 None, 기본값은 None" 이라는 뜻.
- `description: str | None = None` → 선택 필드 (안 넘기면 None)
- `title: str` → 필수 필드
