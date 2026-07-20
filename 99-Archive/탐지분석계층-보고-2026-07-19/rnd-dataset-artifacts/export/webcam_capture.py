"""
웹캠 촬영 도구 — 직접 라벨링용 사진을 맥북 카메라로 모은다.

라벨링에 좋은 데이터 = 다양성: 각도·거리·자세·배경을 바꿔가며 여러 장.
찍은 사진은 한 폴더에 번호순(shot_000.jpg…)으로 저장 → 바로 labelme로 열 수 있음.

조작: 스페이스=촬영 / q 또는 ESC=종료
사용:
  OUT=~/Desktop/pajama-photos python webcam_capture.py
  (첫 실행 시 맥 '카메라 접근 허용' 팝업 — 허용해야 동작)
"""
import os
from pathlib import Path

import cv2

OUT = Path(os.environ.get("OUT", "~/Desktop/labeling-photos")).expanduser()
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ 웹캠을 열 수 없습니다 — 시스템 설정 > 개인정보 보호 > 카메라에서 터미널 허용 확인")

existing = sorted(OUT.glob("shot_*.jpg"))
n = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
print(f"저장 폴더: {OUT}  (기존 {len(existing)}장, 다음 번호 {n})")
print("스페이스=촬영 / q=종료. 각도·거리·자세를 바꿔가며 10~20장 권장.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    cv2.putText(view, f"saved: {n}  [space]=shot  [q]=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("capture (space=shot, q=quit)", view)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord(" "):
        path = OUT / f"shot_{n:03d}.jpg"
        cv2.imwrite(str(path), frame)     # 오버레이 없는 원본 저장
        print(f"  저장: {path.name}")
        n += 1

cap.release()
cv2.destroyAllWindows()
print(f"완료 — 총 {n}장. 다음: ~/rnd-env/bin/labelme {OUT}")
