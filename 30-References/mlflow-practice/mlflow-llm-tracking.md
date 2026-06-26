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

| run | 데이터 | 에포크 | train_loss | val_loss | **간격(val−train)** | 판정 |
|---|---|---|---|---|---|---|
| the-verdict | 5K토큰 | 10 | 0.57 | 6.37 | **5.80** | 심한 과적합(암기) |
| tinyshakespeare | 338K | 1 | 5.39 | 5.84 | 0.45 | 일반화 |
| tinyshakespeare | 338K | 3 | 4.08 | 4.85 | 0.77 | 일반화(약한 과적합 시작) |

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

## 다음 — Colab에서 MLflow
- GPT-2 가중치 **파인튜닝**은 TF 필요 → Colab. MLflow도 Colab서 동작.
- ⚠️ Colab `mlruns`는 휘발성 → **Google Drive 마운트** 후 `set_tracking_uri("file:/content/drive/MyDrive/mlruns")` 로 영구 저장. UI는 포트 연결(`output.serve_kernel_port_as_window`) 또는 Drive mlruns를 로컬로 가져와 봄.

## 관련 노트
- [[quickstart-notes]] — MLflow Iris quickstart(기초 3동사)
- [[../mlops]] — MLOps·MLflow 개념 정본
- [[../../10-Projects/llm-from-scratch/llm-ch5-pretrain]] — 5장(이 학습의 모델·loss 토대)
