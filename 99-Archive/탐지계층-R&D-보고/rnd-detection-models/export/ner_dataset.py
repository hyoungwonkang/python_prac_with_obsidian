"""
NER 데이터 준비 — KLUE-NER 로드 + BERT subword 라벨 정렬(-100).

라벨링 방법론(R&D 주제)의 NER 판:
- KLUE-NER = 한국어 개체명 인식 공개 데이터 (HuggingFace `datasets`).
- 문자 단위로 BIO 태깅되어 있음. 각 문자를 하나의 '단어'로 넣어(is_split_into_words) 라벨과
  1:1 정렬한다. 한글은 대개 **문자 1개 = 토큰 1개**라 이 경우 -100은 주로 [CLS]/[SEP]/패딩에 붙는다.
  다만 한 문자가 여러 subword로 쪼개지는 경우(숫자·영문 등)를 대비해, **첫 subword에만 라벨을 주고
  나머지 subword는 -100**으로 채우는 표준 정렬 안전장치를 둔다(손실에서 제외).
  → 교재 6·7장에서 배운 ignore_index(-100)가 그대로 재등장.

주의: KLUE 공식 test셋은 라벨이 비공개(리더보드용)라, 여기서는 validation 스플릿을
        평가(테스트)용으로 사용한다.

단독 실행 시 데이터 통계를 출력한다:
    ~/rnd-env/bin/python ner_dataset.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from functools import lru_cache

from datasets import load_dataset


@lru_cache(maxsize=1)
def load_klue_ner():
    """KLUE-NER 로드 → (dataset, label_list, label2id, id2label)."""
    ds = load_dataset("klue/klue", "ner")   # datasets 5.x: 스크립트형 "klue"→데이터 repo "klue/klue"
    label_list = ds["train"].features["ner_tags"].feature.names  # ['B-DT','I-DT',...,'O']
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    return ds, label_list, label2id, id2label


def align_labels(examples, tokenizer, max_length):
    """토큰화 + subword 정렬. 첫 subword=원래 라벨, 나머지=-100."""
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        aligned = []
        for wid in word_ids:
            if wid is None:            # [CLS]/[SEP]/패딩
                aligned.append(-100)
            elif wid != prev:          # 새 단어의 첫 subword
                aligned.append(labels[wid])
            else:                      # 같은 단어의 이어지는 subword
                aligned.append(-100)
            prev = wid
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


def build_tokenized(tokenizer, max_length=128, subset=None):
    """train/validation을 토큰화·정렬해 반환. subset이면 앞 N개만(스모크용)."""
    ds, label_list, label2id, id2label = load_klue_ner()
    train = ds["train"]
    val = ds["validation"]
    if subset:
        train = train.select(range(min(subset, len(train))))
        val = val.select(range(min(max(subset // 5, 1), len(val))))
    cols = train.column_names
    train_tok = train.map(lambda x: align_labels(x, tokenizer, max_length),
                          batched=True, remove_columns=cols)
    val_tok = val.map(lambda x: align_labels(x, tokenizer, max_length),
                      batched=True, remove_columns=cols)
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return train_tok, val_tok, label_list, label2id, id2label


if __name__ == "__main__":
    ds, label_list, label2id, id2label = load_klue_ner()
    print("스플릿:", {k: len(v) for k, v in ds.items()})
    print(f"라벨 {len(label_list)}종:", label_list)
    print("\n샘플 1건:")
    ex = ds["train"][0]
    print("  tokens :", "".join(ex["tokens"])[:60], "...")
    print("  tags   :", [id2label[t] for t in ex["ner_tags"][:20]], "...")
