"""
labelme JSON → YOLO txt 변환 — 직접 라벨링 파이프라인의 다리.

labelme는 로컬 라벨링 도구 (민감 도메인 데이터를 외부 서버에 올리지 않음 — 웹 도구 불가 요건).
출력이 JSON이므로 YOLO 규약(클래스번호 cx cy w h, 이미지 대비 비율)으로 변환해
datasets-own/<이름>/의 images/·labels/에 배치한다.

원칙: 라벨은 data.yaml의 names 체계에 강제 — 체계 밖 라벨(오타)은 차단 (PII 검증기와 동일 정신).

사용:
  JSONS=~/라벨링작업폴더 DATASET=datasets-own/my-ds SPLIT=train \
    ~/rnd-env/bin/python labelme_to_yolo.py
  (JSONS: labelme가 저장한 *.json 폴더 / DATASET: make_yolo_dataset.py로 만든 뼈대 / SPLIT: train|val)
"""
import json
import os
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
JSONS = Path(os.environ["JSONS"]).expanduser().resolve()
DATASET = Path(os.environ["DATASET"]).expanduser()
if not DATASET.is_absolute():
    DATASET = HERE / DATASET
SPLIT = os.environ.get("SPLIT", "train")


def main():
    names = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):                       # {0: "안경", ...} 형태도 지원
        names = [names[k] for k in sorted(names)]
    name2id = {n: i for i, n in enumerate(names)}

    img_out = DATASET / "images" / SPLIT
    lbl_out = DATASET / "labels" / SPLIT
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    jfiles = sorted(JSONS.glob("*.json"))
    if not jfiles:
        raise SystemExit(f"❌ {JSONS} 에 labelme JSON이 없습니다.")

    n_box, per_class = 0, {}
    for jf in jfiles:
        d = json.loads(jf.read_text(encoding="utf-8"))
        W, H = d["imageWidth"], d["imageHeight"]
        lines = []
        for s in d.get("shapes", []):
            if s.get("shape_type") != "rectangle":
                print(f"  ⚠️ {jf.name}: {s.get('shape_type')} 도형 무시 (사각형만 변환)")
                continue
            label = s["label"]
            if label not in name2id:                  # 체계 강제 — 오타가 데이터셋을 오염시키기 전에 차단
                raise SystemExit(f"❌ 라벨 '{label}'이 data.yaml 체계 {names}에 없음 ({jf.name}) — "
                                 "오타면 labelme에서 수정, 새 클래스면 data.yaml에 먼저 등록.")
            (xa, ya), (xb, yb) = s["points"][:2]      # 두 모서리 (순서 무관)
            x1, x2 = sorted((xa, xb))
            y1, y2 = sorted((ya, yb))
            lines.append(f"{name2id[label]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            per_class[label] = per_class.get(label, 0) + 1
            n_box += 1
        src = (jf.parent / d["imagePath"]).resolve()  # 원본 이미지도 규약 위치로 복사
        stem = src.stem if src.exists() else jf.stem
        if src.exists():
            shutil.copy(src, img_out / src.name)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")

    print(f"변환 완료: 이미지 {len(jfiles)}장 / 박스 {n_box}개 → {DATASET}/(images|labels)/{SPLIT}/")
    print("클래스별:", per_class)
    print(f"다음 단계: DATA={DATASET / 'data.yaml'} ~/rnd-env/bin/python yolo_train_custom.py")


if __name__ == "__main__":
    main()
