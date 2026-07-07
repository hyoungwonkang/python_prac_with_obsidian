# README (Windows) — NER · YOLO · PII 설치·실행

클린 Windows 환경 기준. 모든 명령은 **PowerShell**에서 프로젝트 폴더(예: `C:\work\rnd-console\src`)로 이동 후 실행.

## 0. 사전 요구

- **Python 3.12** (최소 3.10 — ko-pii가 3.10 미만 설치 불가): https://www.python.org/downloads/ 에서 설치, 설치 시 **"Add python.exe to PATH" 체크**
- 인터넷 연결 (첫 실행 시 모델·데이터 자동 다운로드: klue/bert-base 약 425MB, KLUE-NER 약 37MB, yolov8n+COCO128 약 13MB)
- 확인: `py -3.12 --version` → `Python 3.12.x`

## 1. 가상환경 생성 + 의존성 설치 (버전 고정)

```powershell
py -3.12 -m venv rnd-env
.\rnd-env\Scripts\Activate.ps1        # 프롬프트 앞에 (rnd-env) 표시되면 성공
python -m pip install --upgrade pip
pip install -r requirements.txt       # 반드시 requirements.txt로 (버전 고정 조합)
```

- ⚠️ PowerShell에서 Activate가 막히면(실행 정책): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 재시도.
- **NVIDIA GPU가 있는 경우** (없으면 건너뜀 — CPU로 동작):
  ```powershell
  pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu124
  ```

설치 검증:
```powershell
python -c "import torch, transformers, datasets, ultralytics, ko_pii; print('OK', torch.__version__)"
```

## 2. 실행 — NER (⚠️ ner_klue.pt는 이 단계에서 '생성'됩니다)

> **가중치 파일(ner_klue.pt)은 첨부에 포함되지 않습니다** (432MB, 학습으로 생성되는 산출물).
> `finetune_ner.py`가 **끝까지 완주해야 마지막 단계에서 저장**됩니다 — 중간에 멈추면 파일이 안 생깁니다.

```powershell
# (권장) 1단계 — 빠른 스모크: 몇 분 내 ner_klue.pt 생성 → 콘솔 연동부터 확인
$env:NER_SUBSET="200"; $env:EPOCHS="1"; python finetune_ner.py

# 2단계 — 보고 수치 재현 (F1 ~0.70). CPU면 1시간 이상 소요 주의
$env:NER_SUBSET="6000"; $env:EPOCHS="3"; python finetune_ner.py
```

- 완료 표시: 콘솔에 `가중치 저장: ner_klue.pt` 출력 + **스크립트와 같은 폴더**에 파일 생성 (`dir ner_klue.pt`로 확인).
- 스모크(200건)의 F1은 낮게 나오는 게 정상 — 파일 생성·연동 확인용.

## 3. 실행 — YOLO / PII

```powershell
$env:EPOCHS="1"; python yolo_train.py    # → mAP 출력 + yolo_best.pt 생성
python pii_detect.py                      # PII 마스킹 데모 (학습·가중치 불필요, 즉시 동작)
python pii_eval.py                        # PII 커버리지 측정
```

## 4. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `ner_klue.pt` 없음 | **finetune_ner.py를 끝까지 실행해야 생성** (§2). 중간 오류·중단 시 미생성 — 아래 항목들 먼저 해결 |
| `No module named 'ko_pii'` 등 | venv 활성화 상태에서 `pip install -r requirements.txt` 재확인. Python 3.10 미만이면 ko-pii 설치 자체가 실패 |
| KLUE 로드 실패 (`Invalid HF URI 'klue'`) | `datasets` 버전이 5.0.0인지 확인 (`pip show datasets`) — 구버전은 미지원 |
| 다운로드 실패/멈춤 | 방화벽·프록시에서 huggingface.co, github.com 허용 필요 |
| 학습이 너무 느림 | CPU 환경 정상 현상. `NER_SUBSET`·`EPOCHS` 축소(§2의 스모크) 또는 GPU 설치 명령(§1) 사용 |
| `Activate.ps1` 실행 차단 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## 5. 산출물 위치 (전부 스크립트와 같은 폴더 기준)

| 파일 | 생성 주체 | 용도 |
|---|---|---|
| `ner_klue.pt` (스모크 ~432MB) | finetune_ner.py **완주 시** | NER 추론·콘솔 연동용 가중치 |
| `yolo_best.pt` | yolo_train.py | YOLO 추론용 가중치 |
| `yolo_runs\` | yolo_train.py | 학습 로그·주석 이미지 |
| `datasets\coco128\` | yolo_train.py 첫 실행 | COCO128 캐시 |
