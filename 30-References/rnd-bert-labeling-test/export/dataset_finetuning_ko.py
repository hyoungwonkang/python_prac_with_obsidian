"""
한국어 스팸 데이터 준비 — 영어판 dataset_finetuning.py와 동일 로직(라벨인코딩·균형·70:10:20 분할).
출처: HF dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (영어 UCI를 기계번역한 다국어판).
      labels(ham/spam) + text_ko(한국어) 열 사용.
출력: ko/train.csv · ko/validation.csv · ko/test.csv  (열: Label, Text — 영어판과 동일 포맷)

→ 이 CSV는 finetune_bert_spam.py의 SpamDataset이 그대로 읽을 수 있다 (열 이름 동일).
"""
import urllib.request
from pathlib import Path
import pandas as pd

URL = ("https://huggingface.co/datasets/dbarbedillo/"
       "SMS_Spam_Multilingual_Collection_Dataset/resolve/main/data-augmented.csv")
RAW = Path(__file__).resolve().parent / "ko_spam_raw.csv"
OUT_DIR = Path(__file__).resolve().parent / "ko"


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def main():
    if not RAW.exists():
        print("다운로드 중...")
        urllib.request.urlretrieve(URL, RAW)
    df = pd.read_csv(RAW)

    # 한국어 열 + 라벨만 추출 → 영어판과 같은 (Label, Text) 포맷으로
    df = df[["labels", "text_ko"]].rename(columns={"labels": "Label", "text_ko": "Text"})
    df = df.dropna(subset=["Text"])
    df = df[df["Text"].str.strip() != ""]
    print("원본 라벨 분포:", df["Label"].value_counts().to_dict())

    balanced = create_balanced_dataset(df)
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
    print("균형 후:", balanced["Label"].value_counts().to_dict())

    train_df, val_df, test_df = random_split(balanced, 0.7, 0.1)
    OUT_DIR.mkdir(exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=None)
    val_df.to_csv(OUT_DIR / "validation.csv", index=None)
    test_df.to_csv(OUT_DIR / "test.csv", index=None)
    print(f"저장 완료 → {OUT_DIR}")
    print(f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")


if __name__ == "__main__":
    main()
