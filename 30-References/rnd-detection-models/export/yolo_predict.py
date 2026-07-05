"""
YOLO 추론/데모 — 학습된 가중치(yolo_best.pt)로 이미지에서 객체를 탐지한다.

플랫폼 Image Analyzer의 최소 추론기. 샘플 이미지(COCO128 중 1장, 자동 다운로드된
데이터에서 탐색)에 대해 탐지 결과를 콘솔에 출력하고 주석 이미지를 저장한다.

환경변수: WEIGHTS(yolo_best.pt) · SOURCE(이미지 경로 — 없으면 COCO128에서 자동 선택)

실행:
  ~/rnd-env/bin/python yolo_predict.py
  SOURCE=/path/to/image.jpg ~/rnd-env/bin/python yolo_predict.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("WEIGHTS", HERE / "yolo_best.pt"))

device = "mps" if torch.backends.mps.is_available() else (
    "0" if torch.cuda.is_available() else "cpu")


def find_sample():
    """SOURCE 지정이 없으면 다운로드된 COCO128에서 이미지 1장을 찾는다."""
    if os.environ.get("SOURCE"):
        return os.environ["SOURCE"]
    datasets_dir = Path(SETTINGS.get("datasets_dir", "~/datasets")).expanduser()
    imgs = sorted((datasets_dir / "coco128" / "images" / "train2017").glob("*.jpg"))
    if imgs:
        return str(imgs[0])
    raise SystemExit("샘플 이미지를 찾지 못함 — SOURCE 환경변수로 지정하거나 "
                     "먼저 yolo_train.py를 실행해 COCO128을 받으세요.")


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"가중치 없음: {WEIGHTS} — 먼저 yolo_train.py 실행.")
    source = find_sample()
    print(f"device: {device} | weights: {WEIGHTS.name} | source: {source}")

    model = YOLO(str(WEIGHTS))
    results = model.predict(source=source, device=device, save=True,
                            project=str(HERE / "yolo_runs"), name="predict",
                            exist_ok=True, verbose=False)

    r = results[0]
    names = r.names
    print(f"\n=== 탐지 결과: {len(r.boxes)}개 객체 ===")
    for box in r.boxes:
        cls = names[int(box.cls)]
        conf = float(box.conf)
        print(f"   [{cls}] conf={conf:.2f}")
    print(f"\n주석 이미지 저장: {r.save_dir}")


if __name__ == "__main__":
    main()
