"""
OCR 앙상블 실험 — 두 엔진 교차 검증: "합의는 자동 채택, 불일치만 검수".

오답 프로필이 상이(Easy=모양 혼동 / Paddle=끝 글자 절단)하다는 관찰의 활용 실험.
같은 이미지를 두 엔진으로 읽어 단어 단위로 대조:
  합의(같은 단어)   → 자동 채택 (사람 안 봄)
  불일치(다른 단어) → 검수 큐 (사람이 정답으로 교정한다고 가정)
정답 대조로 세 가지를 채점:
  검수 부담 = 불일치 단어 비율 (사람 작업량)
  맹점     = 합의했는데 둘 다 틀린 단어 (이 정책이 놓치는 것 — 조용한 실패 후보)
  정책 정확도 = 합의 채택 + 불일치 인간 교정 후의 단어 정확도

실행:  python ocr_ensemble.py       (rnd-env에서. IMAGES=images 기본, images_pre 가능)
       paddle 격리 환경은 기본 ~/ocr-env — 다른 위치면 OCR_ENV=경로 로 지정.
"""
import difflib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / os.environ.get("IMAGES", "images")
OUT = HERE / "out"
# easy = 이 스크립트를 실행한 파이썬 그대로(venv 이름·위치 무관) / paddle = 격리 venv (윈도우는 Scripts 구조)
OCR_ENV = Path(os.environ.get("OCR_ENV", Path.home() / "ocr-env")).expanduser()
PY = {"easy": Path(sys.executable),
      "paddle": OCR_ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")}


def images() -> list[Path]:
    return sorted(list(IMAGES.glob("*.png")) + list(IMAGES.glob("*.jpg")))


def ocr_all(engine: str) -> dict[str, list[str]]:
    """엔진별 전용 환경 서브프로세스 1회로 폴더 전체 읽기 → {이미지: 단어 목록}."""
    code = (f"import pathlib, ocr_eval\n"
            f"read = ocr_eval.make_reader()\n"
            f"ps = sorted(list(pathlib.Path(r'{IMAGES}').glob('*.png')) + list(pathlib.Path(r'{IMAGES}').glob('*.jpg')))\n"
            f"for p in ps:\n"
            f"    t = ' '.join(read(p).split())\n"
            f"    (pathlib.Path(r'{OUT}') / (p.stem + '.{engine}.txt')).write_text(t, encoding='utf-8')\n")
    r = subprocess.run([str(PY[engine]), "-c", code], cwd=HERE,
                       capture_output=True, text=True,
                       env={**os.environ, "ENGINE": engine})
    if r.returncode != 0:
        raise SystemExit(f"{engine} OCR 실패:\n{r.stderr[-800:]}")
    return {p.stem: (OUT / f"{p.stem}.{engine}.txt").read_text(encoding="utf-8").split()
            for p in images()}


def gold_align(words: list[str], gold: list[str]) -> set[int]:
    """words에서 gold와 정렬상 일치하는 인덱스 집합 (그 단어가 '정답'인지 판정용)."""
    ok = set()
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=words, b=gold).get_opcodes():
        if op == "equal":
            ok.update(range(a1, a2))
    return ok


def main():
    OUT.mkdir(exist_ok=True)
    gold_file = IMAGES / "ground_truth.txt"
    gold = gold_file.read_text(encoding="utf-8").split() if gold_file.exists() else None

    mode = f"평가(정답 {len(gold)}단어 채점)" if gold else "운영(정답 없음 — 합의/불일치만)"
    print(f"세트: {IMAGES.name} {len(images())}장 | 모드: {mode} | 정책: 합의=자동 채택, 불일치=검수")
    easy_all, paddle_all = ocr_all("easy"), ocr_all("paddle")

    tot_agree = tot_review = tot_blind = tot_words = 0
    n_empty = n_full_agree = n_flag = 0
    if gold:
        print(f"\n{'이미지':<13} {'합의':>4} {'검수부담':>8} {'맹점':>4}  {'정책 결과'}")
        print("-" * 64)
    for stem in easy_all:
        e, p = easy_all[stem], paddle_all[stem]
        agree_idx, diffs = [], []
        for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=e, b=p).get_opcodes():
            if op == "equal":
                agree_idx.extend(range(a1, a2))
            else:
                diffs.append(f"{' '.join(e[a1:a2]) or '∅'}→{' '.join(p[b1:b2]) or '∅'}")
        n_agree = len(agree_idx)
        n_review = max(len(e), len(p)) - n_agree             # 불일치(검수 대상) 근사
        tot_agree += n_agree; tot_review += n_review
        tot_words += max(len(e), len(p))

        if gold:                                             # ── 평가 모드: 정답 채점
            e_ok = gold_align(e, gold)
            blind = [e[i] for i in agree_idx if i not in e_ok]   # 합의했는데 정답 아님
            n_blind = len(blind)
            acc = (len(gold) - n_blind) / len(gold)          # 합의 채택 + 불일치 인간 교정 가정
            print(f"{stem:<13} {n_agree:>4} {n_review:>6}개 {n_blind:>4}  "
                  f"단어 정확도 {acc:.0%}" + (f"  ⚠️ 맹점: {blind}" if blind else ""))
            tot_blind += n_blind
        else:                                                # ── 운영 모드: 합의/불일치만
            if not e and not p:
                n_empty += 1                                 # 둘 다 무텍스트 = 합의(자동 통과)
            elif n_review == 0:
                n_full_agree += 1
            else:
                n_flag += 1
                print(f"  검수 ▶ {stem}: easy {len(e)}단어 / paddle {len(p)}단어 — "
                      + ", ".join(diffs[:3]) + (" …" if len(diffs) > 3 else ""))

    print("-" * 64)
    if gold:
        print(f"합계: 합의 {tot_agree}/{tot_words} ({tot_agree/tot_words:.0%} 자동 통과) | "
              f"검수 부담 {tot_review}개 ({tot_review/tot_words:.0%}) | 맹점 {tot_blind}개")
    else:
        n = n_empty + n_full_agree + n_flag
        print(f"문서 단위: 둘 다 무텍스트 {n_empty} + 전부 합의 {n_full_agree} = "
              f"자동 통과 {n_empty+n_full_agree}/{n} ({(n_empty+n_full_agree)/n:.0%}) | "
              f"검수 대상 {n_flag}장 ({n_flag/n:.0%})")
        if tot_words:
            print(f"단어 단위(텍스트 검출분): 합의 {tot_agree}/{tot_words} ({tot_agree/tot_words:.0%})"
                  f" | 검수 부담 {tot_review}개")
        print("※ 운영 모드 — 정답이 없어 맹점·정확도는 측정 불가 (평가 국면에서만 가능)")


if __name__ == "__main__":
    main()
