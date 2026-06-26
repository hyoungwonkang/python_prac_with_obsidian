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

## PyTorch 기초 (autograd)

[[../10-Projects/pytorch-study|pytorch-study]] 진행 중 만난 개념. 환경 정본은 [[pytorch-env-hybrid]].

### `torch.tensor(data, requires_grad=False)`
- PyTorch의 기본 컨테이너. NumPy 배열 + GPU + 자동미분 지원.
- `requires_grad=True`를 주면 PyTorch가 이 텐서에 대한 **연산 그래프를 추적**한다.
- 일반적으로 **학습 대상(가중치 w, 편향 b)에만 True**, 입력/정답에는 False(기본).

### autograd 한눈에
```
forward:  x → (w, b 사용) → 예측 ŷ → 손실 L
backward: ∂L/∂w, ∂L/∂b 를 자동으로 구함 (체인 룰을 라이브러리가 대신 해줌)
```
- PyTorch는 forward 중 모든 연산을 그래프로 기록 → backward 호출 시 거꾸로 미분.
- 두 가지 호출 방식 (등가):
  - `loss.backward()` → `w.grad`, `b.grad`에 결과가 자동으로 쌓임 (대부분의 학습 루프 방식)
  - `torch.autograd.grad(loss, [w, b])` → 결과를 변수로 직접 받음 (값을 보고 싶을 때 편함)

### `torch.sigmoid(z)`
- 시그모이드 함수 σ(z) = 1 / (1 + e^(-z)). 출력 범위 (0, 1) → **이진 분류 확률**로 해석.
- 단, **이진 분류 학습에는 `BCEWithLogitsLoss`가 수치 안정**: 시그모이드+BCE를 한 번에. 따로 쓰면 매우 큰 z에서 log(0) 위험.

### `F.binary_cross_entropy(input, target)`
- 이진 교차 엔트로피 손실. `input`은 **확률**(0~1 범위, 시그모이드 통과한 값)이어야 함.
- 로짓(시그모이드 통과 전)을 그대로 넣으면 안 됨 → 그땐 `F.binary_cross_entropy_with_logits` 사용.

### `torch.autograd.grad(outputs, inputs, retain_graph=False)`
- 명시적으로 ∂outputs/∂inputs를 계산. `.backward()` 안 부르고 값을 직접 받고 싶을 때.
- 반환값은 **튜플** — 입력이 하나여도 `(tensor,)` 형태로 옴.
- `retain_graph=True` — 같은 그래프에서 두 번 이상 grad/backward를 부를 때 필수. 기본은 한 번 쓰고 그래프 메모리 해제.

### `requires_grad` vs `grad` vs `grad_fn` (헷갈리기 쉬움)
| 속성 | 의미 |
|------|------|
| `t.requires_grad` | 이 텐서를 추적하라(True/False 플래그) |
| `t.grad` | **`.backward()` 호출 후** 채워지는 ∂loss/∂t 값 (그 전엔 None) |
| `t.grad_fn` | 이 텐서를 만든 연산(예: `<AddBackward0>`). leaf 텐서는 None |

### 텐서 차원 (스칼라·벡터·행렬·텐서)
- 축(차원) 수로 구분, **전부 "텐서"의 일종**: 0차원 스칼라(`tensor(5.8)`=loss) / 1차원 벡터 / 2차원 행렬(가중치 표) / 3차원+ 텐서.
- `.shape`로 모양 확인. 예: 토큰 임베딩 가중치 `(50257, 768)`. 텐서 = NumPy 배열 + GPU + autograd.

### batch / seq_len (LLM 입력 모양)
- **seq_len** = 한 시퀀스의 토큰 개수(문장 길이). 모델 `context_length`가 상한.
- **batch** = 한 번에 같이 처리하는 시퀀스 개수 (GPU 병렬 → 빠름). 한 배치 안 시퀀스는 길이가 같아야 함(직사각형 텐서 → `max_length`로 자름).
- 흐름: 입력 `(batch, seq_len)` → 임베딩 `(batch, seq_len, emb_dim)` → 로짓 `(batch, seq_len, vocab_size)`.

### `F.cross_entropy(logits, targets)` (다중분류 — LLM 다음 토큰)
- 다음 토큰(어휘 5만 중 하나) 예측 손실 = $-\log(\text{정답 토큰 확률})$ 평균. **직접 계산 불필요(한 줄)** — softmax+log+평균 자동.
- ⚠️ **로짓을 직접** 입력(softmax 미리 X). `logits`=`(N, 클래스수)`, `targets`=`(N,)` **정수 토큰ID**(원-핫 아님).
- LLM은 `logits.flatten(0,1)` / `targets.flatten()`로 (batch×seq_len) 펼쳐 입력. (이진분류는 위 BCE, 다중분류·LLM은 cross_entropy)

### `torch.optim.AdamW(params, lr, weight_decay)` (옵티마이저)
- `optimizer.step()`의 파라미터 업데이트 알고리즘 = **SGD + 모멘텀 + 파라미터별 적응 보폭(=Adam) + 분리된 weight decay(=W, 과적합 방지)**. 트랜스포머 표준.
- `lr`=보폭(학습률), `weight_decay`=정규화 강도. 파라미터당 상태 2개(모멘텀·분산) 저장 → 학습 메모리↑. 흐름: loss→backward(grad)→**AdamW.step()**(업데이트).

### 실습 흐름 (이 프로젝트의 두 파일)
- [[../10-Projects/pytorch-study/logistic.py]] — forward만: 선형결합 → sigmoid → BCE
- [[../10-Projects/pytorch-study/gradient.py]] — 같은 forward + `requires_grad=True` + `grad()`로 ∂L/∂w1, ∂L/∂b 직접 계산

## PyTorch 부록 A 회고

> 교재 Raschka 『밑바닥부터 만들면서 배우는 LLM』 부록 A(A.1~A.10) 학습을 마친 시점의 회고. 본문 1장 진입 전 자기 점검 자리.
> 본문은 학습한 본인이 직접 채운다 — 골격만 두고, 다 채우면 [[../10-Projects/pytorch-study]]의 A.10 체크박스를 `[x]`로 닫는다.

### 가장 큰 깨달음 3가지

1. ai라는 대집합안에 머신러닝이라는 소집합, 그리고 그 안의 딥러닝이라는 집합이 있고 그 안에 파이토치를 사용하는 훈련 모델이 있음
2. 이 훈련 모델은 텍스트, 이미지를 판별하는 bert, ocr의 주요하게 쓰임
3. 훈련 모델을 쓰는데 필요한 자원은 데이터셋이고 그 환경은 CPU와 GPU

### 헤맸던 부분 / 다시 본다면

- 

### 본문 1장 진입 전 자신감 점검

| 항목 | 자신감 (1~5) | 부족하면 어디로 |
|---|---|---|
| 텐서·dtype·shape 다루기 | | A.2, [[../10-Projects/pytorch-study/ch01_tensor_dtypes.py]] |
| autograd로 gradient 직접 계산 | | A.4, [[../10-Projects/pytorch-study/gradient.py]], [[../10-Projects/pytorch-study/backprop_intuition.md]] |
| `nn.Module`로 모델 정의 | | A.5, [[../10-Projects/pytorch-study/neural.py]] |
| `DataLoader`로 배치 단위 학습 | | A.6, [[../10-Projects/pytorch-study/dataloader.py]] |
| forward → loss → backward → step 루프 | | A.7, [[../10-Projects/pytorch-study/training.py]] |
| `state_dict` 저장·로드 | | A.8 |
| 디바이스(`cuda`/`mps`/`cpu`) 자동선택 | | A.9, [[../10-Projects/pytorch-study/ch00_env_check.py]] |

### 관련 노트

- [[../10-Projects/pytorch-study]] — 학습 진도 정본
- [[../10-Projects/llm-from-scratch]] — 마스터 (부록 A → 본문 1~7장)

## 프로젝트 파일

### `requirements.txt`
- 프로젝트에 필요한 **외부 패키지 목록**을 적어놓은 파일.
- `pip install -r requirements.txt` → 목록의 패키지를 한 번에 설치.
- 다른 사람(또는 다른 PC)에서 프로젝트를 클론한 뒤, 같은 환경을 재현하기 위해 사용.
- 버전 고정 예: `fastapi==0.115.0`, 최소 버전 예: `fastapi>=0.115.0`
- `pip freeze > requirements.txt` → 현재 설치된 패키지를 파일로 내보내기.
