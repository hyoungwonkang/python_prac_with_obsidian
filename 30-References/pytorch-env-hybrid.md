# PyTorch 환경 가이드 — 멀티 머신(인텔 맥 · Apple Silicon · Colab) 하이브리드 (정본)

> **🏆 정본 (Single Source of Truth)**
> 사용자 PC의 `~/.claude/CLAUDE.md`와 메모리는 이 파일을 가리키는 포인터일 뿐. 환경 정보 변경은 이 파일에서만 한다.
> 최초 확정: 2026-06-16 · **Apple Silicon(M4 Max) 추가: 2026-06-23**

이 vault에서 PyTorch/ML 작업을 할 때 따르는 환경·코드 표준. 사용자는 **여러 머신**(인텔 맥, Apple Silicon 맥)과 Colab을 오가며 작업하므로, 각 로컬 머신의 제약과 교재의 Colab 환경을 함께 다룬다.

## 🖥️ 로컬 머신 목록

| 머신 | 칩 | 아키텍처 | 로컬 PyTorch 상한 | 비고 |
|------|----|---------|-----------------|------|
| **MacBook Pro 16,1** | Intel Core i7 6C | x86_64 | **2.2.2 (고정)** | 2.3+ 휠 없음 ⚠️ |
| **MacBook Pro 16,5** | **Apple M4 Max** | arm64 | **최신(2.8.0 확인)** | 36GB·MPS 정식 지원 🚀 |

---

## 🍎 Apple Silicon 머신 (MacBook Pro 16,5 / M4 Max) — 메인 로컬

### 하드웨어
| 항목 | 값 |
|------|-----|
| 모델 | MacBook Pro (Mac16,5) |
| 칩 | **Apple M4 Max** (10 성능 + 4 효율 = 14코어) |
| 아키텍처 | **arm64** |
| 메모리 | **36 GB** 통합 메모리 |
| OS | macOS 26.5.1 (Darwin 25.x) |
| **CUDA** | ❌ 불가 (NVIDIA 아님) |
| **MPS** | ✅ 정식 지원 (`is_built=True`, `is_available=True`, `mps:0` 검증 완료) |

### 확정 환경 구성 (2026-06-23 셋업·검증)
| 항목 | 값 |
|------|-----|
| Python | **3.9.6** (시스템 `/usr/bin/python3`, Command Line Tools) |
| 가상환경 | **`~/ml-env`** (venv) — `python3 -m venv ~/ml-env` |
| 활성화 | `source ~/ml-env/bin/activate` (또는 직접 `~/ml-env/bin/python`) |
| PyTorch | **2.8.0** |
| tiktoken | 0.13.0 |
| NumPy | 2.0.2 |
| matplotlib | 3.9.4 |
| 디바이스 | `mps` (가속) — `torch.rand(3, device='mps')` 동작 확인 |

- 인텔 맥의 2.2.2 상한·CUDA·소스빌드 제약은 **이 머신에는 적용되지 않음**. 최신 torch를 그대로 설치.
- 36GB 통합 메모리 + M4 Max → 본문 3·4장 실습 코드는 로컬에서 쾌적. GPT-2 가중치 로드급 무거운 사전훈련은 여전히 Colab T4 권장.
- ⚠️ **Python 3.9 캐비엇**: 시스템 파이썬 3.9.6은 비교적 구버전(상위 EOL 임박). 현재 torch 2.8.0과 호환되어 실습엔 문제없으나, 추후 더 새 파이썬이 필요하면 Homebrew(`brew install python@3.12`)로 재구성 가능. 그 전까지는 3.9.6 `~/ml-env` 재사용. (Homebrew는 이미 설치됨: brew 6.0.3)
- 🔴 **TensorFlow 로컬 불가 (2026-06-25 확인)**: 시스템 파이썬 3.9.6 `~/ml-env`에서 `import tensorflow`가 **C++ 레벨 크래시**(`libc++abi: mutex lock failed: Invalid argument`). macOS CLT 파이썬 빌드 비호환 이슈. → 로컬 고집 시 대안: ① `transformers`로 TF 우회, ② Homebrew `python@3.12` 별도 venv 재구성(미검증). torch/tiktoken 등은 정상.
- 🟢 **GPT-2 가중치 로컬 로드 해결 (2026-06-29)**: 위 대안 ①을 구현. `10-Projects/llm-from-scratch/hf_weight_adapter.py`가 HF `GPT2LMHeadModel.from_pretrained`(PyTorch)로 받아 교재 `GPTModel`에 로드(**TF 불필요**, `USE_TF=0`). 가능 이유: HF Conv1D 가중치 방향이 TF `gpt_download` 형식과 동일 → 기존 `load_weights_into_gpt` 재사용. → **GPT-2 분류 파인튜닝 전체를 M4 Max 로컬에서 완주(1.88분, val acc 97.5%)**. 이제 GPT-2 트랙도 BERT처럼 완전 로컬. (단 OpenAI 원본 TF 체크포인트를 직접 읽는 `gpt_download.py`는 여전히 TF→Colab.)

---

## 🖥️ 인텔 머신 (MacBook Pro 16,1) — 보조 로컬 (제약 多)

### 하드웨어
| 항목 | 값 |
|------|-----|
| 모델 | MacBook Pro 16,1 |
| CPU | Intel Core i7 6-Core 2.6 GHz (**x86_64**) |
| GPU | Intel UHD 630 (내장) / AMD Radeon Pro 5300M 4GB (외장) |
| OS | macOS 15 (Darwin 24.5.0) |
| **CUDA** | ❌ 불가 |
| **MPS** | ✅ 사용 가능 (Metal → Radeon Pro 5300M) |

### 🔴 핵심 제약 (이 인텔 머신에 한함, 반드시 준수)
1. **인텔 맥에는 PyTorch 2.3+ 휠이 존재하지 않음.** 2.3부터 macOS x86_64 지원 중단. 로컬 설치는 **2.2.2가 최대치**.
2. **CUDA 코드 로컬 미동작.** `device='cuda'`는 항상 실패.
3. **소스 빌드 금지.** 의존성 지옥 + 수 시간 + 불안정.

### 확정 환경 구성
| 항목 | 값 |
|------|-----|
| Python | 3.12.13 (Homebrew) |
| 가상환경 | `~/ml-env` (venv) |
| PyTorch | **2.2.2** (인텔 맥 최종) |
| torchvision / torchaudio | 0.17.2 / 2.2.2 |
| NumPy | 1.26.4 (`numpy<2`) |
| 디바이스 | `mps` 또는 `cpu` |

> 두 머신 모두 venv 경로는 `~/ml-env`로 같지만 **물리적으로 다른 기기**이므로 내용물(파이썬·torch 버전)이 다른 게 정상. 각 머신에서 자기 `~/ml-env`를 재사용한다.

---

## ☁️ Colab 환경 (교재 원본 재현·풀 학습)
| 항목 | 값 |
|------|-----|
| PyTorch | **2.6.0** (교재가 테스트된 버전) |
| GPU | NVIDIA T4 (무료 티어) |
| 디바이스 | `cuda` |

## 📚 작업 분담 원칙

| 작업 유형 | 환경 | 이유 |
|-----------|------|------|
| 텐서 기초·autograd·작은 모델·코드 작성/디버그 | **로컬 (M4 Max 우선)** | 빠른 반복, MPS 가속, 최신 torch |
| 본문 3·4장 실습 코드 실행 | **로컬 M4 Max** | 36GB·MPS로 충분 |
| 교재 코드 결과 정확 재현 | **Colab 2.6.0** | 교재와 100% 동일 환경 |
| GPT-2 가중치 로드(분류 FT) | **로컬 M4 Max OK** | `hf_weight_adapter.py`(HF, TF 우회) — 1.88분 |
| GPT-2 원본 TF 체크포인트(`gpt_download`)·대형 사전훈련 | **Colab T4** | TF 필요 / NVIDIA GPU |
| (인텔 맥 사용 시) 텐서 문법·소규모 실험 | 인텔 로컬 2.2.2 | 2.2.2 상한 내에서만 |

## ⚠️ 버전 차이 (작성 시 주의)

### `torch.load()`의 `weights_only` 기본값
2.6부터 `True`가 기본. 양쪽 안전한 작성법:
```python
state = torch.load("model.pt", weights_only=True)
```
M4 Max(2.8.0)·Colab(2.6.0)은 기본 `True`, 인텔 맥(2.2.2)은 기본 `False`이므로 **명시 권장**.

### NumPy 1.x ↔ 2.x
M4 Max는 NumPy 2.0.2, 인텔 맥은 1.26.4(`numpy<2`). 대부분 호환되나 일부 deprecated API 차이 가능 — 문제 시 버전 명시.

### 그 외 핵심 API는 모두 동일
`nn.Module`, `nn.Linear/Conv2d/LSTM`, `optim.*`, `DataLoader/Dataset`, `transforms`, `autograd`, `.backward()`, `.grad` 등 교재 기본 API는 전 환경 동일 동작.

## ✅ 코드 작성 표준 (전 환경 호환)

```python
import torch

# 디바이스 자동 선택 (로컬/Colab 모두 OK)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 모델 로드 호환성 (전 버전 안전)
state = torch.load("model.pt", weights_only=True)

# MPS 일부 연산자 미지원 시 자동 fallback (필요 시)
# export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 🚫 하지 말아야 할 것

- ❌ (인텔 맥) `pip install torch>=2.3` 시도 — 반드시 실패
- ❌ `device='cuda'` 하드코딩
- ❌ `conda` 사용 (venv로 충분)
- ❌ (인텔 맥) 소스 빌드 시도

## 🔄 갱신 조건

이 문서는 **다음 경우에만 갱신**한다:
- ✅ ~~Apple Silicon 맥으로 교체~~ → **2026-06-23 M4 Max 추가 완료** (이 갱신)
- 새 로컬 머신 추가 / 머신 환경 버전 변경
- 교재 또는 PyTorch 버전 요구사항 변경
- 사용자가 명시적으로 환경 변경을 요청

## 🔗 관련 노트

- [[../10-Projects/pytorch-study]] — 이 환경을 사용하는 학습 프로젝트
- [[../10-Projects/llm-from-scratch]] — LLM 본문 트랙 (로컬 실습 환경 사용)
- [[python-basics]] — Python 문법·라이브러리 기초
