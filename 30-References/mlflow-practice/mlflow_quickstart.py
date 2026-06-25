"""MLflow 공식 quickstart 연습 (Iris + LogisticRegression).

핵심 3동사 익히기: log_params / log_metric / log_model.
개념 정본: [[../mlops.md]]

실행 전 별도 터미널에서 UI 서버를 먼저 띄운다:
    ~/ml-env/bin/mlflow ui --port 5000
그러면 아래 set_tracking_uri 가 그 서버(localhost:5000)로 직접 기록 →
브라우저 localhost:5000 새로고침하면 run 이 바로 보인다.
(서버 없이 로컬 파일로만 기록하려면 set_tracking_uri 를 "file:./mlruns" 로 바꾸면 됨)
"""
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── 2단계: 훈련 데이터 준비 ──
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# ── 3단계: 모델 학습 + 평가 (평범한 sklearn) ──
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도(accuracy): {accuracy:.4f}")

# ── 4단계: MLflow로 실험 기록 ──
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # ★ 켜둔 UI 서버에 직접 기록
mlflow.set_experiment("MLflow Quickstart")        # 실험(묶음) 이름

with mlflow.start_run():                            # 한 번의 실험 실행(run)
    mlflow.log_params(params)                       # ① 하이퍼파라미터
    mlflow.log_metric("accuracy", accuracy)         # ② 성능 지표
    mlflow.set_tag("Training Info", "Iris LR 기본 모델")  # 메모 태그

    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(          # ③ 모델 저장
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
    )
    print("run 기록 완료. UI 새로고침 →", model_info.model_uri)

# ── 5단계: 저장한 모델 다시 불러와 예측 (재현성 확인) ──
loaded = mlflow.pyfunc.load_model(model_info.model_uri)
print("불러온 모델 예측(앞 5개):", loaded.predict(X_test[:5]))
print("실제 정답(앞 5개):      ", y_test[:5])
