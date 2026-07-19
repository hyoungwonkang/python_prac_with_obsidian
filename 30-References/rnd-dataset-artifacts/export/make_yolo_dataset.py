"""
YOLO 커스텀 데이터셋 규약 스캐폴더 — "이 폴더 모양대로 채우면 학습기가 그대로 받는다".

데이터셋 규약(폴더 계약):
  <루트>/<NAME>/
    data.yaml            ← 클래스 목록 + 경로 (이 파일이 데이터셋의 정본 선언)
    images/train/ *.jpg  ← 학습 이미지
    images/val/   *.jpg  ← 검증 이미지
    labels/train/ *.txt  ← 이미지와 같은 이름, 줄마다: 클래스번호 cx cy w h (0~1 정규화)
    labels/val/   *.txt

두 모드:
  기본       — datasets-own/<NAME>/ 에 빈 뼈대 생성 (직접 라벨링용 → git 추적 자산.
               라벨링 도구: labelImg, Roboflow, CVAT 등에서 YOLO 형식으로 내보내 채움)
  DEMO=1     — datasets/yolo-demo/ 에 합성 도형(원/사각형) 이미지+정답 라벨을 자동 생성
               (규약이 실제로 학습되는지 end-to-end 증명용 — 재생성 가능이라 git 제외)

실행 예:
  NAME=my-first-labels CLASSES=사람,차량 python make_yolo_dataset.py
  DEMO=1 python make_yolo_dataset.py
"""
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = os.environ.get("DEMO", "0") == "1"
NAME = os.environ.get("NAME", "yolo-demo" if DEMO else "my-dataset")
CLASSES = [c.strip() for c in os.environ.get(
    "CLASSES", "circle,square" if DEMO else "class0").split(",")]
ROOT = HERE / ("datasets" if DEMO else "datasets-own") / NAME


def write_yaml():
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (ROOT / "data.yaml").write_text(
        f"# YOLO 데이터셋 정본 선언 — make_yolo_dataset.py 생성\n"
        f"path: {ROOT}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8")


def scaffold():
    for split in ("train", "val"):
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    write_yaml()
    (ROOT / "README.md").write_text(
        f"# {NAME}\n\n클래스: {CLASSES}\n\n"
        "- `images/train|val/`에 이미지, `labels/train|val/`에 같은 이름의 `.txt` 라벨.\n"
        "- 라벨 형식: `클래스번호 cx cy w h` (0~1 정규화, 줄당 객체 1개).\n"
        "- labelImg·Roboflow·CVAT에서 'YOLO 형식'으로 내보내면 이 규약과 일치한다.\n"
        "- 학습: `DATA=<이 폴더>/data.yaml python yolo_train_custom.py`\n",
        encoding="utf-8")
    print(f"뼈대 생성 → {ROOT}/ (클래스 {len(CLASSES)}개: {CLASSES})")


def make_demo():
    import cv2
    import numpy as np
    random.seed(123)
    counts = {"train": 16, "val": 6}
    for split, n in counts.items():
        img_dir, lbl_dir = ROOT / "images" / split, ROOT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            size = 480
            img = np.full((size, size, 3),
                          random.randint(30, 90), dtype=np.uint8)
            noise = np.random.default_rng(i).integers(
                0, 25, (size, size, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            lines = []
            for _ in range(random.randint(1, 3)):     # 이미지당 도형 1~3개
                cls = random.randint(0, 1)            # 0=circle, 1=square
                half = random.randint(30, 70)
                cx = random.randint(half + 5, size - half - 5)
                cy = random.randint(half + 5, size - half - 5)
                color = tuple(random.randint(150, 255) for _ in range(3))
                if cls == 0:
                    cv2.circle(img, (cx, cy), half, color, -1)
                else:
                    cv2.rectangle(img, (cx - half, cy - half),
                                  (cx + half, cy + half), color, -1)
                w = h = 2 * half / size
                lines.append(f"{cls} {cx/size:.6f} {cy/size:.6f} {w:.6f} {h:.6f}")
            cv2.imwrite(str(img_dir / f"{split}_{i:03d}.jpg"), img)
            (lbl_dir / f"{split}_{i:03d}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: 이미지 {n}장 + 라벨 {n}개")
    write_yaml()
    print(f"합성 데모 데이터셋 생성 → {ROOT}/")


if __name__ == "__main__":
    make_demo() if DEMO else scaffold()
