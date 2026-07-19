"""
YOLO 학습 — ultralytics YOLOv8n 전이학습 (COCO128 스모크).

플랫폼의 "Image Analyzer(이미지 위험 분류)" 계층에 대응하는 최소 객체탐지 실습.
사전학습 가중치(yolov8n.pt)에서 전이학습 = BERT 파인튜닝과 같은 발상(무거운 본체
재사용, 소량 데이터로 미세조정). COCO128은 128장짜리 공식 스모크 데이터로,
가중치·데이터 모두 첫 실행 시 자동 다운로드된다.

환경변수: EPOCHS(3) · IMGSZ(640) · DEVICE(자동: mps→cpu) · MODEL(yolov8n.pt)

실행:
  ~/rnd-env/bin/python yolo_train.py
  EPOCHS=1 ~/rnd-env/bin/python yolo_train.py     # 빠른 스모크
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

EPOCHS = int(os.environ.get("EPOCHS", 3))
IMGSZ = int(os.environ.get("IMGSZ", 640))
MODEL = os.environ.get("MODEL", "yolov8n.pt")
HERE = Path(__file__).resolve().parent

if os.environ.get("DEVICE"):
    device = os.environ["DEVICE"]
else:
    device = "mps" if torch.backends.mps.is_available() else (
        "0" if torch.cuda.is_available() else "cpu")


def main():
    print(f"device: {device} | model: {MODEL} | epochs: {EPOCHS} | imgsz: {IMGSZ}")
    model = YOLO(MODEL)                       # 사전학습 가중치 로드(자동 다운로드)

    results = model.train(
        data="coco128.yaml",                 # 128장 공식 스모크셋(자동 다운로드)
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device,
        project=str(HERE / "yolo_runs"),
        name="coco128",
        exist_ok=True,
        verbose=True,
    )

    # 성능 지표(mAP) 출력
    metrics = model.val(data="coco128.yaml", device=device,
                        project=str(HERE / "yolo_runs"), name="val", exist_ok=True)
    print("\n=== YOLO 평가 (COCO128) ===")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")

    # best.pt를 export 폴더로 복사(재사용·추론용)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, HERE / "yolo_best.pt")
        print(f"가중치 저장: yolo_best.pt (from {best})")


if __name__ == "__main__":
    main()
