"""
비식별화 — YOLO 검출 + OpenCV 블러/모자이크.

플랫폼의 개인정보 보호·채증 비식별화 요구(아키텍처 §개인정보 필수사항)의 최소 구현.
cv2 5.0 빌드에 objdetect(Haar) 모듈이 없어 얼굴 검출은 불가 → 이미 검증한 YOLO로
'사람' 영역을 검출하고, 그 영역만 OpenCV로 흐리게 처리한다. (검출=YOLO, 처리=OpenCV
역할 분담. NER·YOLO R&D의 직접 연결.)

파라미터(환경변수):
  SOURCE : 입력 이미지 (없으면 ultralytics zidane.jpg — 사람 2명)
  TARGET : 비식별 대상 COCO 클래스 (기본 'person')
  MODE   : blur(가우시안) | mosaic(모자이크) (기본 blur)

실행:
  python opencv_deidentify.py
  SOURCE=/path/img.jpg MODE=mosaic python opencv_deidentify.py
산출: opencv_out/deid/deidentified.png (+ 원본 대비)
사전: 인터넷 최초 1회(yolov8n.pt 자동 캐시) 또는 로컬 캐시 존재.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
OUT = HERE / "opencv_out" / "deid"
TARGET = os.environ.get("TARGET", "person")
MODE = os.environ.get("MODE", "blur")


def find_source():
    src = os.environ.get("SOURCE")
    if src and Path(src).exists():
        return src
    import ultralytics
    sample = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    if sample.exists():
        return str(sample)
    raise SystemExit("입력 이미지를 찾지 못함 — SOURCE 환경변수로 지정하세요.")


def blur_roi(img, x1, y1, x2, y2, mode):
    """박스 영역만 비식별 처리해 되붙인다."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if mode == "mosaic":
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // 16), max(1, h // 16)),
                           interpolation=cv2.INTER_LINEAR)
        img[y1:y2, x1:x2] = cv2.resize(small, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
    else:  # gaussian blur — 커널을 영역 크기에 비례해 강하게
        k = max(15, (min(roi.shape[:2]) // 2) | 1)  # 홀수 보장
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else (
        "0" if torch.cuda.is_available() else "cpu")

    source = find_source()
    img = cv2.imread(source)
    print(f"입력: {source} | device: {device} | 대상: {TARGET} | 모드: {MODE}")

    model = YOLO("yolov8n.pt")  # 사전학습(COCO 80클래스) — 자동 캐시
    r = model.predict(source=source, device=device, verbose=False)[0]
    names = r.names

    count = 0
    for box in r.boxes:
        if names[int(box.cls)] != TARGET:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        blur_roi(img, x1, y1, x2, y2, MODE)
        count += 1

    out = OUT / "deidentified.png"
    cv2.imwrite(str(out), img)
    print(f"검출된 '{TARGET}': {count}개 → 비식별 처리 완료")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
