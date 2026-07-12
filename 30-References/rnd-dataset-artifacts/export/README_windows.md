# README (Windows) — 학습 데이터 산출물 환경 설치·실행

클린 Windows 환경 기준. 모든 명령은 **PowerShell**에서 이 폴더(export)로 이동 후 실행.

## 0. 사전 요구

- **Python 3.12** (최소 3.10 — ko-pii 요건): 설치 시 **"Add python.exe to PATH" 체크**
- 인터넷 연결 (첫 실행 시 자동 다운로드: klue/bert-base 약 425MB, yolov8n 약 6MB)
- 확인: `py -3.12 --version` → `Python 3.12.x`

## 1. 가상환경 + 의존성 설치 (버전 고정)

```powershell
py -3.12 -m venv rnd-env
.\rnd-env\Scripts\Activate.ps1        # 프롬프트 앞에 (rnd-env) 표시되면 성공
python -m pip install --upgrade pip
pip install -r requirements.txt       # 반드시 requirements.txt로 (버전 고정 조합)
```

- ⚠️ Activate가 막히면: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 재시도.
- **NVIDIA GPU가 있는 경우** (없으면 건너뜀 — CPU로 동작, 스크립트가 자동 감지):
  ```powershell
  pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu124
  ```

설치 검증:
```powershell
python -c "import torch, transformers, ultralytics, ko_pii; print('OK', torch.__version__)"
```

## 2. 실행 — 텍스트 분류 (환경변수는 PowerShell `$env:` 문법)

```powershell
# 새 분류기: text,label CSV만 준비하면 끝 (스모크: 몇 분 내 완료)
$env:DATA="C:\path\to\my_data.csv"; $env:NAME="my-clf"; python train_text.py
# 산출물 재사용 분류
$env:ARTIFACT="artifacts\my-clf"; python classify_text.py "분류할 문장"
```

## 3. 실행 — 직접 라벨링 → YOLO 학습 (지시 4′ 파이프라인)

```powershell
# ① 뼈대 생성
$env:NAME="my-ds"; $env:CLASSES="안경,지갑"; python make_yolo_dataset.py
# ② 라벨링 (로컬 GUI — 이미지가 외부로 나가지 않음)
labelme C:\사진폴더
# ③ labelme JSON → YOLO 규약 변환 (라벨 오타는 자동 차단)
$env:JSONS="C:\사진폴더"; $env:DATASET="datasets-own\my-ds"; $env:SPLIT="train"; python labelme_to_yolo.py
# ④ 학습 → 산출물 3종 세트
$env:DATA="datasets-own\my-ds\data.yaml"; $env:EPOCHS="40"; python yolo_train_custom.py
```

## 4. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `No module named 'ko_pii'` 등 | venv 활성화 후 `pip install -r requirements.txt` 재확인. Python 3.10 미만이면 ko-pii 설치 실패 |
| 학습이 너무 느림 | CPU 정상 현상 — `LIMIT`·`EPOCHS` 축소(스모크) 또는 §1의 GPU 설치 명령 |
| labelme 창이 안 뜸 | venv 활성화 상태인지 확인 (`.\rnd-env\Scripts\labelme.exe` 직접 실행도 가능) |
| `❌ 데이터가 너무 적습니다` | 정상 가드 — 단독 학습은 수십 건 필요, 소량 검수분은 기존 train에 편입 |
| 다운로드 멈춤 | 방화벽·프록시에서 huggingface.co, github.com 허용 |

## 5. 산출물 위치

| 경로 | 내용 |
|---|---|
| `artifacts\<이름>\` | 3종 세트 (model.pt + label_map.json + meta.json) — 재사용의 단위 |
| `datasets-own\<이름>\` | 직접 라벨링 데이터 (재생성 불가 자산 — 보존 대상) |
| `mlflow.db` | 실험 기록 (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |
