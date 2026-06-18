# llm-ch2-text

[[llm-from-scratch]] 마스터 플랜의 **Phase 2 / 2장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 2장. 텍스트 데이터 다루기.

## 개요

원시 텍스트를 토큰화하고 임베딩으로 바꿔 모델 입력 파이프라인을 만드는 단계.

## 체크리스트

- [x] 2.1 토크나이저 (단어 단위 → 서브워드 → BPE) — `tokenization.py`(V1), `tokenizerV2.py`(특수 토큰)
- [x] 2.2 `tiktoken`으로 GPT-2 BPE 사용 실습 — `tokenizerV2.py`
- [x] 2.3 토큰 임베딩 + 위치(positional) 임베딩 — `gpt_embedding.py`
- [x] 2.4 슬라이딩 윈도우 데이터셋·`DataLoader` 구현 — `gpt_datasampling.py`
- [x] 2.5 본인 텍스트로 동일 파이프라인 재현 — `gpt_datasampling.py` (tinyshakespeare)

## 실습 코드

- `tokenization.py` — `SimpleTokenizerV1`: 정규식 분리 → vocab(`str_to_int`/`int_to_str`) → encode/decode.
- `tokenizerV2.py` — `SimpleTokenizerV2`: 미등록 토큰 `<|unk|>`·문서 경계 `<|endoftext|>` 추가. 이어서 `tiktoken` GPT-2 BPE로 OOV 없이 인코딩.
- `gpt_datasampling.py` — `GPTDatasetV1` + `create_dataloader_v1`: 슬라이딩 윈도우로 (입력, 타깃) 쌍 생성. 입력=`token_ids[i:i+max_length]`, 타깃=`token_ids[i+1:i+max_length+1]`(한 칸 시프트).
- `gpt_embedding.py` — `nn.Embedding`으로 토큰 임베딩 + 위치 임베딩을 더해 최종 입력 임베딩(`token + pos`) 구성.

## 배운 것 (핵심 정리)

- **입력-타깃 쌍**은 `+1` 시프트가 만든다(다음 토큰 예측). `stride`는 쌍을 만드는 게 아니라 **다음 쌍을 어디서 시작할지**(간격·겹침)를 정한다.
- **stride 트레이드오프**: `stride < max_length`면 윈도우가 겹쳐 예제 多·중복↑(데이터 적을 때 유리), `= max_length`면 겹침·빠짐 없이 한 번씩(정석), `> max_length`면 토큰을 건너뛰어 데이터 낭비. GPT 사전훈련처럼 데이터가 방대하면 `stride = max_length` 관례.
- **토큰 임베딩 vs 위치 임베딩**: 토큰 임베딩은 *무슨 단어인지*(결정론적·위치 독립적 → 재현성 좋음)만 담고 *몇 번째인지*는 못 담는다. self-attention(3장)도 순서를 모르므로 **위치 임베딩을 더해** 순서를 주입한다. 최종 입력 = 토큰 임베딩 + 위치 임베딩.
- **재현 실습 메모(tinyshakespeare)**: stride=1 + 1.1MB 전문은 텐서 67만 개를 메모리에 올려(~810MB) 스와핑 유발. 입력 텍스트를 2만 자로 줄이고(`input.txt` 19,985바이트, 장면 경계 컷) 다운로드는 파일 없을 때만 받도록(`os.path.exists` 가드) 개선 → 실행 10.2s→2.15s, 메모리 810MB→201MB.

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.
✅ 로컬 ml-env에서 4개 파일 정상 동작 확인(2026-06-18). 입력 임베딩 shape `[8, 4, 256]` 재현.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[llm-ch3-attention]] — 다음 장(위치 정보를 쓰는 self-attention)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
