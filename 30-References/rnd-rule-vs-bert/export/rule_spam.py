"""
룰 기반 스팸 분류기 — '분류 잘하는 법' R&D의 베이스라인.

원칙 (테스트 누출 금지):
  - 키워드는 train.csv의 스팸/정상 출현 통계로만 선정 (스팸 출현 ≥8회, 정상 출현 0회 상위권에서 수작업 선별)
  - 문턱(threshold)은 validation.csv로 조정
  - test.csv는 규칙을 만드는 동안 열지 않는다 — 시험문제를 미리 보면 비교가 무효

사용:
  from rule_spam import classify          # classify(text) -> 0(정상)/1(스팸)
  python rule_spam.py       # validation으로 문턱 1~3 채점 (문턱 결정 근거 출력)
"""
import re

# train 1,045건 통계에서 선별 (스팸에서만 나오는 단어들). 근거: 스팸출현/정상출현
KEYWORDS = [
    "상을",        # 39/0 — "상을 받으려면"
    "현금",        # 34/0
    "모바일",      # 29/0
    "노키아",      # 27/0 — 번역 스팸 특유의 경품 폰
    "파운드",      # 25/0 — 영국 SMS 스팸 번역 잔재
    "보너스",      # 22/0
    "보증",        # 19/0
    "긴급",        # 15/0
    "휴가",        # 18/0 — "무료 휴가 당첨"
    "전화하십시오", # 14/0 — 명령형 유도
    "기다리고",    # 20/0 — "상품이 기다리고 있습니다"
]

PATTERNS = [
    re.compile(r"\d{5,}"),                                   # 5자리+ 숫자 — 단축번호·전화 유도
    re.compile(r"£"),                                        # 통화 기호 (파운드)
    re.compile(r"(?i)\b(txt|stop|call|free|win|winner)\b"),  # 미번역 영어 스팸 어휘
    re.compile(r"(?i)https?://|www\."),                      # URL
]


def score(text: str) -> int:
    """맞은 신호의 개수 (키워드 + 패턴)."""
    t = str(text)
    s = sum(1 for k in KEYWORDS if k in t)
    s += sum(1 for p in PATTERNS if p.search(t))
    return s


# validation.csv 149건으로 문턱 1~3을 채점해 결정 (아래 __main__ 출력이 근거)
THRESHOLD = 1


def classify(text: str, threshold: int = THRESHOLD) -> int:
    return 1 if score(text) >= threshold else 0


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    HERE = Path(__file__).resolve().parent
    val = pd.read_csv(HERE / "../../rnd-bert-labeling-test/export/ko/validation.csv")
    golds = val["Label"].tolist()
    print(f"validation {len(val)}건으로 문턱 조정 (test는 열지 않음)")
    for th in (1, 2, 3):
        preds = [classify(t, th) for t in val["Text"]]
        print(f"  문턱 {th}: 정확도 {accuracy_score(golds, preds):.4f} / "
              f"스팸P {precision_score(golds, preds, zero_division=0):.4f} / "
              f"스팸R {recall_score(golds, preds, zero_division=0):.4f} / "
              f"스팸F1 {f1_score(golds, preds, zero_division=0):.4f}")
