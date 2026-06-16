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
├── ch01_basics.py             ← 텐서 기초 (디바이스 자동선택 동작 확인)
└── notebooks/                 ← Colab 노트북 백업 (.ipynb)
```

## 단계 (Phase)

### Phase 0 — 환경 셋업
- [x] Homebrew Python 3.12 + `~/ml-env` venv 생성
- [x] 로컬 PyTorch 2.2.2 설치 + MPS 동작 확인 ([[../30-References/pytorch-env-hybrid]])
- [x] 레포 클론 + vault에 프로젝트 등록
- [ ] Colab 새 노트북 + GPU 활성화 + 환경 검증 셀
- [ ] Colab → vault 동기화 워크플로 결정 (수동 다운로드 vs `gh` API)

### Phase 1 — 텐서 기초
- [x] `ch01_basics.py` — 텐서 생성·연산, 디바이스 자동선택 (로컬 동작 확인됨)
- [ ] dtype·shape·broadcasting·indexing 실습
- [ ] CPU ↔ MPS 이동(`.to(device)`) 성능 차이 측정
- [ ] [[../30-References/python-basics]]에 PyTorch 기초 개념 정리 시작

### Phase 2 — autograd & 첫 학습 루프
- [ ] `requires_grad`, `.backward()`, `.grad` 개념 실습
- [ ] 선형 회귀 from-scratch (loss/optimizer 없이)
- [ ] `nn.Module` + `optim.SGD` 도입
- [ ] 동일 코드 Colab(2.6.0)·로컬(2.2.2) 결과 비교

### Phase 3 — 첫 분류 모델 (MNIST)
- [ ] `torchvision.datasets.MNIST` + `DataLoader`
- [ ] MLP → CNN 단계적 도입
- [ ] 학습/평가 루프 구조화 (train/eval mode, no_grad)
- [ ] **Colab T4에서 풀 학습, 로컬은 코드 작성·디버그용**

### Phase 4 — 검증·재현성·실험 관리
- [ ] random seed 고정, `torch.backends.cudnn.deterministic`
- [ ] 체크포인트 저장/로드 (`weights_only=True` 명시)
- [ ] 실험 결과를 [[../90-Daily]] 데일리 노트에 기록하는 흐름 만들기

### Phase 5 — 응용 (선택)
- [ ] Transfer learning (사전학습 모델 fine-tune)
- [ ] [[rec-planner]]와 접점: 임베딩 추출 등

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
