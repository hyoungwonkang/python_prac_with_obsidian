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

### Alembic
- DB 스키마 마이그레이션 도구. SQLAlchemy와 함께 사용.
- 모델(Python 클래스)이 바뀌면, Alembic이 그 차이를 감지해서 DB 테이블을 자동으로 맞춰줌.
- `alembic revision --autogenerate` → 변경 감지해서 마이그레이션 파일 생성
- `alembic upgrade head` → 마이그레이션 적용 (테이블 생성/수정)
- SQLite → Postgres 전환 시에도 동일한 마이그레이션 파일로 양쪽 적용 가능.

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
