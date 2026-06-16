# pytorch-study

PyTorch 입문 학습 프로젝트. 교재(Colab/PyTorch 2.6.0 기준)를 따라 실습하면서, **인텔 맥 로컬 환경 한계와 Colab GPU를 병행**하는 하이브리드 워크플로를 굳히는 게 목표.

## Context

- [[todo-app]](완료)·[[rec-planner]](진행 중) 다음 트랙: **머신러닝 입문**.
- 사용자는 Python·FastAPI·LLM 앱 개발 경험은 있지만, PyTorch·딥러닝은 첫 학습.
- 교재가 Colab(PyTorch 2.6.0 + NVIDIA T4) 기준 → 로컬은 보조, Colab이 메인.
- 환경 제약·코드 표준의 정본은 [[../30-References/pytorch-env-hybrid]] 한 곳.

## 기술 스택

| 구분 | 선택 | 이유 |
|---|---|---|
| 학습 프레임워크 | **PyTorch** | 교재 지정 |
| 로컬 버전 | **PyTorch 2.2.2** (확정) | 인텔 맥 x86_64 지원 최종 버전 — 2.3+ 휠 없음 |
| 로컬 가상환경 | `~/ml-env` (venv) | 단일 venv 재사용 (새로 만들지 않음) |
| 로컬 가속 | MPS (Radeon Pro 5300M) | CUDA 불가 → MPS fallback |
| 클라우드 메인 | **Google Colab + PyTorch 2.6.0 + T4 GPU** | 교재 환경과 100% 일치 |
| 코드 동기화 | 이 vault (`10-Projects/pytorch-study/`) | git이 단일 진실 공급원, Colab도 이 폴더에서 작업 |

## 환경 제약 (요약)

자세한 내용은 [[../30-References/pytorch-env-hybrid]] 정본 참조. **요약**:

- ❌ 로컬에 PyTorch 2.3+ 설치 시도 금지 (인텔 맥 휠 없음)
- ❌ `device='cuda'` 하드코딩 금지
- ✅ 항상 `cuda → mps → cpu` 자동선택 패턴
- ✅ `torch.load()` 사용 시 `weights_only=True` 명시 (2.2/2.6 호환)

## 폴더 구조

```
10-Projects/pytorch-study/
├── README.md
├── requirements.txt           ← 로컬 2.2.2 환경용
├── ch00_env_check.py          ← 디바이스 자동선택 + 환경 검증 (로컬 mps + Colab cuda 양쪽 통과)
├── ch01_tensor_dtypes.py      ← 텐서 dtype·shape 첫 실습
├── logistic.py                ← forward: 선형 → sigmoid → BCE
├── gradient.py                ← autograd로 ∂loss/∂w1, ∂loss/∂b 직접 계산
└── notebooks/
    └── ch01_basics.ipynb      ← Colab에서 GitHub에 직접 커밋, 첫 셀에 Colab 배지
```

## 단계 (교재 부록 A 절 + 선행 셋업)

> 출처: Raschka, *밑바닥부터 만들면서 배우는 LLM* 부록 A. 각 절을 교재 따라 진행하면서 필요한 실습 코드는 직접 작성한다.

### 선행: 환경 셋업 ✅ 완료 (2026-06-16)
- [x] Homebrew Python 3.12 + `~/ml-env` venv 생성
- [x] 로컬 PyTorch 2.2.2 설치 + MPS 동작 확인 ([[../30-References/pytorch-env-hybrid]])
- [x] 레포 클론 + vault에 프로젝트 등록
- [x] Colab 새 노트북 + T4 GPU 활성화 + 환경 검증 셀 (PyTorch 2.6.0+cu124, Tesla T4)
- [x] Colab → vault 동기화 워크플로 확정 — Colab "파일 → GitHub에 사본 저장" → 로컬 merge

### A.1 파이토치란 무엇인가요?
- [ ] 교재 본문 학습
- [ ] 핵심 메모 정리 → [[../30-References/python-basics]]

### A.2 텐서 이해하기
- [ ] 교재 본문 학습
- [ ] 실습 코드 — dtype·shape·broadcasting·indexing

### A.3 모델을 계산 그래프로 보기
- [ ] 교재 본문 학습
- [ ] 실습 코드 — forward 그래프 시각화

### A.4 자동 미분을 손쉽게
- [ ] 교재 본문 학습
- 작성한 실습 코드:
	- `logistic.py` — 선형 → sigmoid → BCE forward 한 스텝
	- `gradient.py` — autograd로 ∂loss/∂w1, ∂loss/∂b 직접 계산 (2026-06-16: `(-0.0898, -0.0817)`)
	- [[../30-References/python-basics]]에 PyTorch autograd 섹션 신설

### A.5 다층 신경망 만들기
- [ ] 교재 본문 학습
- [ ] 실습 코드 — `nn.Module` 서브클래싱

### A.6 효율적인 데이터 로더 설정하기
- [ ] 교재 본문 학습
- [ ] 실습 코드 — `Dataset` / `DataLoader`

### A.7 일반적인 훈련 루프
- [ ] 교재 본문 학습
- [ ] 실습 코드 — forward → loss → backward → optimizer.step

### A.8 모델 저장과 로드
- [ ] 교재 본문 학습
- [ ] 실습 시 `torch.save(state_dict)` + `torch.load(..., weights_only=True)`

### A.9 GPU로 훈련 성능 최적화하기
- [ ] 교재 본문 학습
- [ ] CPU ↔ MPS 이동(`.to(device)`) 성능 차이 측정 (로컬)
- [ ] 같은 코드를 Colab cuda에서 실행해 결과·속도 비교

### A.10 요약
- [ ] 부록 A 전체 회고 메모 → [[../30-References/python-basics]]

## 지금까지 만든 실습 코드 (절 매핑)

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `ch00_env_check.py` | 디바이스 자동선택, 양 환경 결과 일치 확인 | 선행 / A.1 |
| `logistic.py` | 선형 → sigmoid → BCE forward 한 스텝 | A.4 부근 |
| `gradient.py` | autograd로 ∂loss/∂w1, ∂loss/∂b 직접 계산 | A.4 |
| `notebooks/ch01_basics.ipynb` | Colab 환경 검증 노트북 | 선행 |

## 작업 분담 원칙

| 작업 | 환경 |
|------|------|
| 텐서 문법·autograd·소규모 실험 | 로컬 (`~/ml-env`, MPS) |
| 교재 코드 결과 정확 재현 | Colab 2.6.0 |
| MNIST/CIFAR-10 풀 학습 | Colab T4 GPU |
| 코드 작성·디버그 | 로컬 (빠른 반복) |

## 검증 방법

- `ch01_basics.py` 직접 실행 → 디바이스/버전 출력 확인 (현재 통과 ✅)
- 같은 .py 파일을 Colab에 업로드 실행 → `cuda` 선택 + 결과 동일 확인
- 각 Phase 종료 시점에 [[../30-References/python-basics]] 갱신

## 참고

- [[../30-References/pytorch-env-hybrid]] — 환경 정본 (제약·근거·코드 표준)
- [[../30-References/python-basics]] — 파이썬 문법·라이브러리 기초 정리 (PyTorch 개념도 여기 추가)
