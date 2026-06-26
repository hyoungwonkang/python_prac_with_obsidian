---
title: MLflow 실전 — LLM 학습 train/val 추적·비교
tags: [reference, mlflow, mlops, llm]
---

# MLflow 실전 — LLM 학습 train/val 추적·비교

> Iris quickstart([[quickstart-notes]]) 다음 단계: **실제 LLM 학습(5장)에 MLflow 적용.**
> 코드: `10-Projects/llm-from-scratch/train_with_mlflow.py` (로컬 from-scratch). 개념 정본: [[../mlops]].

## 무엇을 했나
랜덤 초기화 GPT를 the-verdict/tinyshakespeare로 학습하며 **하이퍼파라미터(log_params) + train/val loss(log_metric, step별) + 모델(log_model)** 을 MLflow에 기록 → UI에서 **run끼리 비교**. (TF 불필요 → 로컬 mps. mlruns는 실행 폴더에 영구 저장.)

Iris의 3동사를 그대로 쓰되 `log_metric`에 **`step=global_step`** 만 추가하면 train/val 곡선이 그려진다.

## 핵심 실험 — 데이터 크기·에포크 → 일반화

| run | 시작 | 데이터 | 에포크 | 시작 train | 끝 train/val | 판정 |
|---|---|---|---|---|---|---|
| 로컬 from-scratch | 랜덤 | the-verdict 5K | 10 | ~9.8 | 0.57 / 6.37 | 심한 과적합(암기) |
| 로컬 from-scratch | 랜덤 | tinyshakespeare 338K | 1 | ~6 | 5.39 / 5.84 | 일반화 |
| 로컬 from-scratch | 랜덤 | tinyshakespeare | 3 | — | 4.08 / 4.85 | 일반화(약한 과적합 시작) |
| **Colab 파인튜닝** | **GPT-2** | tinyshakespeare | 1(~100step) | **4.0** ⭐ | 3.48 / 3.39 | **시작부터 낮음 = 사전지식** |

> **핵심: from-scratch는 시작 loss ~10(엉망), 파인튜닝은 시작 ~4.0**(GPT-2가 이미 영어 앎). 파인튜닝 생성도 더 매끄러움("KING RICHARD II: I have heard your counsels, and I am sure"). = 사전지식 재사용의 위력. (간격 비교: train↔val 간격 작을수록 일반화)

### 읽는 법 — 핵심은 절댓값이 아니라 **train↔val 간격**
- **the-verdict**: train만 0.57로 폭락, val 6.37 정체 → 간격 5.8 = **외움**(작은 데이터).
- **tinyshakespeare**: train·val이 **붙어서 같이 내려감**(에포크 1→3에 val 5.84→4.85↓) → **일반화**(큰 데이터).
- 간격이 작을수록 일반화 잘 됨. 학습량 달라도(에포크 수 차이) **간격은 robust한 비교 지표.**

## 배운 개념

### train_loss 높은 게 꼭 나쁜 건 아님
tinyshakespeare 1에포크 train 5.39(높음)는 "외우지 않은 건강한 상태"(1에포크+66배 데이터). the-verdict의 train 0.57(낮음)이 오히려 **암기**라 나쁜 신호.

### "오래 학습" ≠ "많이 배움"
- **오래 학습** = 같은 데이터 여러 번(에포크↑, 시간 축). 작은 데이터로 오래 → **암기·과적합**(시간 써도 실력 안 늘어남).
- **많이 학습** = 더 많고 다양한 데이터(양 축) → **일반화**(진짜 배움).
- → 반복보다 **다양한 데이터**가 실력(일반화)을 만든다. (GPT-2가 수십억 토큰 = 많이 배워 새 텍스트에도 통함)

### 실행 방식 구분
- **오래 학습** = `num_epochs`↑ (한 실행에서 데이터 더 여러 번).
- **파일 재실행** = 매번 **랜덤 초기화로 처음부터**(별개 run, 이어 학습 아님). 같은 설정이면 결과 거의 동일(중복).
- **이어 학습(resume)** = 체크포인트 `load_state_dict`로 복원해 계속 (현재 train_with_mlflow.py엔 없음).

### MLflow 비교의 가치
값(params)을 바꿔 돌릴 때마다 run이 쌓이고, UI에서 **여러 run의 loss 곡선을 겹쳐** "어떤 설정이 val을 낮추나"를 비교 → 콘솔 print로는 못 하는 것(2장 Train & Tune).

## 실행 (로컬)
```bash
cd 10-Projects/llm-from-scratch
~/ml-env/bin/python train_with_mlflow.py     # TRAIN/GPT_CONFIG 값 바꿔가며
~/ml-env/bin/mlflow ui                        # 같은 폴더 → localhost:5000
```

## Colab 실전 — GPT-2 파인튜닝 + MLflow (2026-06-26)

코드: `10-Projects/llm-from-scratch/finetune_gpt2_mlflow.py` (weight_load + train_with_mlflow 통합).
GPT-2 가중치 로드(TF)는 Colab 전용 → 거기서 tinyshakespeare 파인튜닝, mlruns는 **개인 Drive**에 저장.

### ⚠️ Colab MLflow 운영 노하우 (겪은 함정)
- **`MLFLOW_ALLOW_FILE_STORE=true` 필수**: Colab의 새 MLflow가 file store(`file:.../mlruns`)를 기본 거부(maintenance mode). store에 닿는 **모든 곳**에서 필요:
  - 학습 스크립트 → `os.environ.setdefault(...)` 내장.
  - **새 셀의 `search_runs`/조회** → 셀 맨 위에 `os.environ["MLFLOW_ALLOW_FILE_STORE"]="true"`.
  - `mlflow ui` → `MLFLOW_ALLOW_FILE_STORE=true mlflow ui ...`.
- **Drive 마운트로 영구 저장**: `drive.mount('/content/drive')` → `set_tracking_uri("file:/content/drive/MyDrive/mlruns")`. (마운트 실패 `credential propagation unsuccessful` = 서드파티 쿠키 차단 → 허용)
- **GPU 런타임 확인**: `device: cpu`면 매우 느림 → 런타임 유형 변경 T4 GPU(`device: cuda`).
- **UI 보기**: Colab 포트 프록시는 `Invalid Host header`(MLflow DNS rebinding 방지)로 막힘 → **(권장) Drive mlruns를 로컬로 가져와 `mlflow ui`**, 또는 UI 없이 `mlflow.search_runs(...)`로 run 표 비교.
- **첫 실행**: GPT-2 124M 가중치(498MB) 다운로드 + 런타임 바꾸면 재다운로드.

### UI 없이 run 비교 (Colab서 바로)
```python
import os; os.environ["MLFLOW_ALLOW_FILE_STORE"]="true"
import mlflow; mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
print(mlflow.search_runs(experiment_names=["gpt2-finetune"])[
    ["params.dataset","params.num_epochs","params.learning_rate",
     "metrics.final_train_loss","metrics.final_val_loss"]])
```

## 관련 노트
- [[quickstart-notes]] — MLflow Iris quickstart(기초 3동사)
- [[../mlops]] — MLOps·MLflow 개념 정본
- [[../../10-Projects/llm-from-scratch/llm-ch5-pretrain]] — 5장(이 학습의 모델·loss 토대)
