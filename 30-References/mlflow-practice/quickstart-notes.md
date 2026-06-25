---
title: MLflow Quickstart 학습 노트
tags: [reference, mlflow, mlops]
---

# MLflow Quickstart 학습 노트

> 공식 quickstart(Iris + LogisticRegression)을 따라가며 익힌 내용. 실습 코드: `mlflow_quickstart.py`.
> 상위 개념: [[../mlops]] · 병행 트랙: [[../../10-Projects/llm-from-scratch/llm-ch5-pretrain]]

개념 학습 원칙: **단계마다 직전 위에 "새로운 것"을 보강(누적)** — 같은 걸 반복하지 않고 한 칸씩 새 능력을 더한다.

## 단계별 요약 (2~6단계)

| 단계 | 핵심 | 새로 보강된 것 |
|---|---|---|
| 2 데이터 준비 | sklearn Iris 로드·`train_test_split`·`params` 딕셔너리 | 토대 |
| 3 autolog | `mlflow.sklearn.autolog()` | **한 줄 자동 기록** |
| 4 UI 보기 | `mlflow ui`/`mlflow server` | **run을 웹에서 확인** |
| 5 수동 로깅 | `with start_run(): log_*` | **무엇을 기록할지 제어** |
| 6 로드·추론 | `load_model` → `predict` | **저장 모델 재사용(배포 토대)** |

## 핵심 개념

### scikit-learn(sklearn)
- 파이썬 **고전(전통) 머신러닝 라이브러리**. `LogisticRegression`·결정트리·`train_test_split`·`accuracy_score` 등 제공.
- 딥러닝(PyTorch, LLM 트랙)과 대비. Iris는 작아서 sklearn 로지스틱 회귀로 충분.

### autolog vs 수동 로깅
- **autolog**: `autolog()` 한 줄 → `fit()`을 **가로채(후킹)** params·metrics·model·메타데이터를 **자동** 기록. 표준 라이브러리에 편함.
- **수동(3동사)**: `log_params`/`log_metric`/`log_model`을 **직접** 호출. 장점은 결과가 달라 보이는 게 아니라 **"무엇을 기록할지 내가 고르는 제어권"**.
- 화면에서 차이 확인: 수동 run이 autolog run보다 정보 많음(직접 지정한 `iris_model` + 테스트 평가 `Eval` 데이터셋까지).

### run은 누적된다 (덮어쓰지 않음)
- **`fit()` 1회 = run 1개**(autolog 기준). MLflow는 run을 **쌓아** 비교용 히스토리를 만든다.
- autolog 켠 채 `fit()` + 수동 `start_run` 블록이 같이 있으면 **1 실행 = run 2개**.
- run 이름(`glamorous-cub-797` 등)은 자동 부여 고유 식별자.

### ⭐ 위치 일치 원칙 (가장 자주 막힌 부분)
- 스크립트도 UI도 기본 저장소가 **실행한 폴더의 `./mlruns`**.
- **스크립트 실행 폴더 = UI(`mlflow ui`/`server`) 실행 폴더** 여야 run이 보인다. 다르면 UI에 "Default만" 보임.
- run을 만드는 건 **`python ...py`(쓰기)**, 보여주는 건 **`mlflow server`(읽기)** — 별개 명령. 서버만 띄운다고 run이 생기지 않음.
- `mlflow ui` ≈ `mlflow server`(로컬 동일), **기본 포트 5000**(`--port` 생략 가능). `mlflow ui`가 최단.
- venv는 `source ~/ml-env/bin/activate` 후 `mlflow`/`python` 짧게 사용. (`ml-env`는 환경 폴더일 뿐, 코드 위치 아님)

### 모델 로드·추론 (6단계) — flavor
- `model_info.model_uri`(5단계 산출) → `load_model`로 복원 → `predict`. **저장→로드→예측**으로 사이클 닫힘 = 재현·배포 토대.
- **flavor = 모델을 불러오는 형식**(같은 모델을 여러 형식으로 저장):
  - `mlflow.pyfunc.load_model` — **범용**(PDF 같음). 프레임워크 무관 `predict()` 통일, 배포용.
  - `mlflow.sklearn.load_model` — **네이티브**(.docx 같음). sklearn 고유 기능까지.

### 예측 결과 확인
- `pd.DataFrame(X_test, columns=...)` + `actual_class`/`predicted_class` 열 → 입력·정답·예측 비교 표.
- ⚠️ `result[:4]`만으론 **Jupyter는 자동 표시**, **`.py` 스크립트는 안 나옴 → `print(result[:4])`** 필요.
- 이 표는 **로컬 출력일 뿐 MLflow엔 자동 기록 안 됨**. UI에서 보려면 `mlflow.log_table(...)` 또는 CSV `log_artifact`로 **직접 기록**(start_run 블록 안).
- MLflow UI에 자동으로 있는 건 **accuracy(지표)·params·model**. 개별 예측은 직접 로깅해야 보임.

## 실행 방법 (정리)
```bash
# 1) venv 활성화
source ~/ml-env/bin/activate
# 2) 연습 폴더로 이동 (= mlruns 생길 곳)
cd 30-References/mlflow-practice
# 3) 스크립트 실행 (run 생성)
python mlflow_quickstart.py
# 4) UI 서버 (같은 폴더에서 — run 표시), localhost:5000
mlflow ui
```

## 관련 노트
- [[../mlops]] — MLOps 개념 정본
- [[../../10-Projects/llm-from-scratch/llm-ch5-pretrain]] — LLM 5.2 학습 루프에 MLflow 통합(커스텀 loss는 수동 log_metric)
- [[../_References]] — 레퍼런스 인덱스
