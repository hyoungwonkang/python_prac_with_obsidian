# PyTorch 환경 가이드 — 인텔 맥 + Colab 하이브리드 (정본)

> **🏆 정본 (Single Source of Truth)**
> 사용자 PC의 `~/.claude/CLAUDE.md`와 메모리는 이 파일을 가리키는 포인터일 뿐. 환경 정보 변경은 이 파일에서만 한다.
> 최종 확정: 2026-06-16

이 vault에서 PyTorch/ML 작업을 할 때 따르는 환경·코드 표준. 사용자의 인텔 맥 한계와 교재의 Colab 환경을 함께 다룬다.

## 🖥️ 하드웨어 (변경 불가 조건)

| 항목 | 값 |
|------|-----|
| 모델 | MacBook Pro 16,1 |
| CPU | Intel Core i7 6-Core 2.6 GHz (**x86_64**) |
| GPU 내장 | Intel UHD Graphics 630 |
| GPU 외장 | AMD Radeon Pro 5300M 4GB |
| OS | macOS 15 (Darwin 24.5.0) |
| **CUDA** | ❌ **불가** (NVIDIA GPU 아님, macOS는 NVIDIA 드라이버 미지원) |
| **MPS** | ✅ 사용 가능 (Metal API → Radeon Pro 5300M) |

## 🔴 핵심 제약 (반드시 준수)

1. **인텔 맥에는 PyTorch 2.3+ 휠이 존재하지 않음.**
   - PyTorch는 2.3부터 macOS x86_64 지원을 완전히 중단.
   - PyPI에 `torch-2.X.X-*-macosx_*_x86_64.whl` 휠이 없음 (2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9 전부 확인).
   - 로컬 설치는 **2.2.2가 최대치**이며, 그 이상 시도하지 말 것.
2. **CUDA 관련 코드는 로컬에서 동작하지 않음.** `device='cuda'`는 항상 실패.
3. **소스 빌드 시도 금지.** 의존성 지옥 + 수 시간 소요 + 불안정. 가치 없음.

## 🟢 확정된 환경 구성

### 로컬 환경 (텐서 문법·소규모 실험·코드 작성)
| 항목 | 값 |
|------|-----|
| Python | **3.12.13** (Homebrew 설치) |
| 가상환경 | **`~/ml-env`** (venv, 단일) — 새로 만들지 않음 |
| 활성화 | `source ~/ml-env/bin/activate` |
| PyTorch | **2.2.2** (인텔 맥 최종) |
| torchvision | 0.17.2 |
| torchaudio | 2.2.2 |
| NumPy | 1.26.4 (`numpy<2`) |
| 디바이스 | `mps` (가속) 또는 `cpu` (fallback) |

### Colab 환경 (교재 원본 재현·풀 학습)
| 항목 | 값 |
|------|-----|
| PyTorch | **2.6.0** (교재가 테스트된 버전) |
| GPU | NVIDIA T4 (무료 티어) |
| 디바이스 | `cuda` |

## 📚 작업 분담 원칙

| 작업 유형 | 환경 | 이유 |
|-----------|------|------|
| 텐서 기초, autograd, 작은 모델 | **로컬 2.2.2** | 빠른 반복, MPS 가속 |
| 교재 코드 결과 정확 재현 | **Colab 2.6.0** | 교재와 100% 동일 환경 |
| MNIST/CIFAR-10 등 소규모 비전 | 로컬 가능 | MPS로 충분 |
| MNIST 풀 학습/CIFAR-10/대형 모델 | **Colab** | NVIDIA GPU 필요 |
| 코드 작성/디버그 (반복 多) | 로컬 | Colab 세션 초기화 비용 회피 |

## ⚠️ 2.2.2 ↔ 2.6.0 차이 (작성 시 주의)

### 단 하나의 의미 있는 breaking change
**`torch.load()`의 `weights_only` 기본값이 2.6부터 `True`로 변경**.

```python
# 양쪽에서 안전한 작성법 (권장)
state = torch.load("model.pt", weights_only=True)
```

### 그 외 핵심 API는 모두 동일
`nn.Module`, `nn.Linear/Conv2d/LSTM`, `optim.SGD/Adam/AdamW`,
`DataLoader/Dataset`, `transforms`, `autograd`, `.backward()`, `.grad` 등
교재의 모든 기본 API는 양쪽에서 동일하게 동작.

## ✅ 코드 작성 표준 (양 환경 호환)

```python
import torch

# 디바이스 자동 선택 (로컬/Colab 모두 OK)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 모델 로드 호환성
state = torch.load("model.pt", weights_only=True)

# MPS 일부 연산자 미지원 시 자동 fallback (필요 시)
# export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 🚫 하지 말아야 할 것

- ❌ `pip install torch==2.4.0` / `2.5.0` / `2.6.0` 시도 (반드시 실패)
- ❌ `device='cuda'` 하드코딩
- ❌ `conda` 사용 (Homebrew + venv로 충분)
- ❌ 새 가상환경 만들기 (`~/ml-env` 그대로 재사용)
- ❌ 소스 빌드 시도

## 🔄 갱신 조건

이 문서는 **다음 경우에만 갱신**한다 (그 외 변경 금지):
- 사용자가 Apple Silicon 맥으로 교체 → x86_64 제약 해제, 전면 재작성
- 교재 또는 PyTorch 버전 요구사항 변경
- 사용자가 명시적으로 환경 변경을 요청

## 🔗 관련 노트

- [[../10-Projects/pytorch-study]] — 이 환경을 사용하는 학습 프로젝트
- [[python-basics]] — Python 문법·라이브러리 기초 (PyTorch 개념도 점진 추가)
