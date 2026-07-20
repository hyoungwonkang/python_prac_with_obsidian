"""
YOLO 커스텀 데이터 범용 학습기 — data.yaml 규약만 지키면 어떤 데이터셋이든 같은 방식으로 학습.

입력 계약: DATA = make_yolo_dataset.py 규약의 data.yaml 경로
산출물 규약(텍스트 학습기와 동일한 3종 세트) — artifacts/<NAME>/:
  - best.pt         학습된 가중치 (ultralytics 체크포인트)
  - label_map.json  클래스 번호↔이름 (data.yaml names에서 추출)
  - meta.json       베이스모델·파라미터·mAP50·MLflow run id

실행 예:
  DEMO=1 python make_yolo_dataset.py          # 합성 데모 데이터 생성
  DATA=datasets/yolo-demo/data.yaml EPOCHS=5 python yolo_train_custom.py
"""
import json
import os
import shutil
from pathlib import Path

import mlflow
import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

DATA = Path(os.environ.get("DATA", HERE / "datasets/yolo-demo/data.yaml"))
if not DATA.is_absolute():
    DATA = HERE / DATA
BASE = os.environ.get("BASE", "yolov8n.pt")
EPOCHS = int(os.environ.get("EPOCHS", 40))   # 소량 데이터 기준 — 몇 에폭으론 신규 클래스를 못 배움
IMGSZ = int(os.environ.get("IMGSZ", 320))
NAME = os.environ.get("NAME", DATA.parent.name)
EXPERIMENT = os.environ.get("EXPERIMENT", "YOLO-커스텀")
OUT_DIR = HERE / "artifacts" / NAME

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# MLflow 3.x: 파일 저장소(./mlruns)는 유지보수 모드 → sqlite로 명시 고정.
# ultralytics가 프로세스 내에서 추적 URI를 바꿔도 우리 기록은 항상 이 DB로 간다.
MLFLOW_DB = f"sqlite:///{HERE / 'mlflow.db'}"


def main():
    if not DATA.exists():
        raise SystemExit(f"❌ data.yaml 없음: {DATA} — make_yolo_dataset.py로 먼저 생성")
    names = yaml.safe_load(DATA.read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):                       # list/dict 둘 다 허용
        names = dict(enumerate(names))
    print(f"데이터: {DATA}\n클래스 {len(names)}개: {names} / device={DEVICE}")

    model = YOLO(BASE)
    model.train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE,
                project=str(HERE / "yolo_runs"), name=NAME, exist_ok=True,
                verbose=False)
    best = HERE / "yolo_runs" / NAME / "weights" / "best.pt"

    metrics = YOLO(str(best)).val(data=str(DATA), imgsz=IMGSZ, device=DEVICE,
                                  project=str(HERE / "yolo_runs"),
                                  name=f"{NAME}-val", exist_ok=True)
    map50 = float(metrics.box.map50)
    print(f"검증 mAP50: {map50:.4f}")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.log_params({"베이스모델": BASE, "에폭": EPOCHS,
                           "이미지크기": IMGSZ, "클래스수": len(names),
                           "데이터": str(DATA)})
        mlflow.log_metric("mAP50", map50)
        run_id = run.info.run_id

    # ---- 산출물 3종 세트 저장 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUT_DIR / "best.pt")
    (OUT_DIR / "label_map.json").write_text(
        json.dumps({"id2label": {str(k): v for k, v in names.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"name": NAME, "task": "object-detection",
                    "base_model": BASE, "imgsz": IMGSZ,
                    "num_labels": len(names), "data": str(DATA),
                    "metrics": {"mAP50": round(map50, 4)},
                    "mlflow_run_id": run_id},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"산출물 저장 → {OUT_DIR}/ (best.pt + label_map.json + meta.json)")


if __name__ == "__main__":
    main()
