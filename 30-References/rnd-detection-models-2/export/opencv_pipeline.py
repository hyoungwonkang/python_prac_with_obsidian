"""
전처리 파이프라인 (end-to-end) — 프레임 → YOLO 검출 → OpenCV 비식별 → 저장.

플랫폼 아키텍처 "② 추출/전처리 → ③ 1차 탐지(이미지)"를 하나로 잇는 미니 파이프라인.
동영상 증적을 프레임으로 분해하고, 각 프레임에서 사람을 검출해 비식별 처리한 뒤
저장 — OpenCV(전처리·비식별)와 YOLO(검출)의 결합 데모.

입력:
  VIDEO 환경변수 지정 시 그 영상을 프레임 분해. 없으면 ultralytics 번들 샘플
  2장(bus.jpg=사람 여럿, zidane.jpg=사람 2명)을 '프레임 묶음'으로 사용 —
  외부 파일 없이도 검출→비식별이 실제로 동작하도록.

파라미터: VIDEO · EVERY(기본 15) · TARGET(기본 person) · MODE(blur|mosaic)
실행:  ~/rnd-env/bin/python opencv_pipeline.py
산출:  opencv_out/pipeline/deid_XXXX.png + 콘솔 프레임별 검출 수
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path

import cv2

from opencv_deidentify import blur_roi  # 검출 영역 비식별 로직 재사용

HERE = Path(__file__).resolve().parent
OUT = HERE / "opencv_out" / "pipeline"
EVERY = int(os.environ.get("EVERY", "15"))
TARGET = os.environ.get("TARGET", "person")
MODE = os.environ.get("MODE", "blur")


def iter_frames():
    """(인덱스, 프레임) 생성 — VIDEO 있으면 분해, 없으면 샘플 2장."""
    video = os.environ.get("VIDEO")
    if video and Path(video).exists():
        cap = cv2.VideoCapture(video)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % EVERY == 0:
                yield idx, frame
            idx += 1
        cap.release()
        return
    import ultralytics
    assets = Path(ultralytics.__file__).parent / "assets"
    for i, name in enumerate(["bus.jpg", "zidane.jpg"]):
        p = assets / name
        if p.exists():
            yield i, cv2.imread(str(p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else (
        "0" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n.pt")
    print(f"device: {device} | 대상: {TARGET} | 모드: {MODE}")

    total_frames = total_hits = 0
    for idx, frame in iter_frames():
        r = model.predict(source=frame, device=device, verbose=False)[0]
        names = r.names
        hits = 0
        for box in r.boxes:
            if names[int(box.cls)] != TARGET:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            blur_roi(frame, x1, y1, x2, y2, MODE)
            hits += 1
        cv2.imwrite(str(OUT / f"deid_{idx:04d}.png"), frame)
        print(f"  프레임 {idx:04d}: '{TARGET}' {hits}개 비식별")
        total_frames += 1
        total_hits += hits

    print(f"\n처리 프레임 {total_frames} | 비식별 {TARGET} 총 {total_hits}개 → {OUT}")


if __name__ == "__main__":
    main()
