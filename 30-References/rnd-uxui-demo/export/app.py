"""
통합 UXUI 데모 — 지시 3. 완료된 R&D 모듈 5개를 탭 2개짜리 데모 UI로 통합 (제작, 신규 학습 없음).

  텍스트 탭: BERT 스팸 분류(지시 1 산출물 3종 세트) + RULE·하이브리드(지시 2)
             + PII 마스킹(ko-pii) + NER 개체 추출(ner_klue.pt)
  이미지 탭: YOLO 박스(yolov8n) + KoCLIP 한국어 상황 태그(지시 4 — 프롬프트 실시간 편집)

실행:
  ~/rnd-env/bin/python app.py    # → http://127.0.0.1:7860
모델은 첫 사용 시 로딩(지연 로딩) — 탭별 첫 응답만 수 초 걸림.
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import torch

HERE = Path(__file__).resolve().parent
REFS = HERE.parent.parent                                  # 30-References/
sys.path.insert(0, str(REFS / "rnd-rule-vs-bert/export"))   # rule_spam
sys.path.insert(0, str(REFS / "rnd-detection-models/export"))  # predict_ner, ner_dataset

SPAM_ART = REFS / "rnd-dataset-artifacts/export/artifacts/ko-spam-full"
YOLO_PT = REFS / "rnd-detection-models/export/yolov8n.pt"
KOCLIP = "Bingsu/clip-vit-base-patch32-ko"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

NER_KO = {"PS": "인명", "LC": "지명", "OG": "기관", "DT": "날짜", "TI": "시간", "QT": "수량"}
# coco128 전수 실측(2026-07-12)으로 보강: 스포츠·실내 후보가 없으면 해당 장면이
# 표류하거나 도박 후보로 흘러감(주방 30.7%→실내 97.7%, 야구 33%→스포츠 99%).
# ※ 데모 기본값일 뿐 — 실전 세트는 자체/도박 도메인 이미지에서 재확정 필요.
DEFAULT_PROMPTS = ("온라인 도박 사이트 화면\n카드나 도박 게임을 하는 사람들\n일상 생활 사진\n"
                   "집 안이나 실내 공간 사진\n스포츠 경기 사진\n음식 사진\n거리와 차량 사진\n"
                   "동물 사진\n문서나 서류")

_cache = {}   # 모델 지연 로딩 (탭별 첫 사용 시 1회)


def get_spam():
    if "spam" not in _cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        meta = json.loads((SPAM_ART / "meta.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            meta["base_model"], num_labels=meta["num_labels"])
        model.load_state_dict(torch.load(SPAM_ART / "model.pt", map_location="cpu"))
        _cache["spam"] = (tok, model.to(DEVICE).eval(), meta)
    return _cache["spam"]


def get_ner():
    if "ner" not in _cache:
        import predict_ner
        _cache["ner"] = predict_ner.load()   # (tokenizer, model, id2label)
    return _cache["ner"]


def get_pii():
    if "pii" not in _cache:
        from ko_pii import Anonymizer, ProcessingMode
        _cache["pii"] = Anonymizer(mode=ProcessingMode.STRICT, strategy="tokenize")
    return _cache["pii"]


def get_yolo():
    if "yolo" not in _cache:
        from ultralytics import YOLO
        _cache["yolo"] = YOLO(str(YOLO_PT))
    return _cache["yolo"]


def get_koclip():
    if "koclip" not in _cache:
        from transformers import CLIPModel, CLIPProcessor
        _cache["koclip"] = (CLIPModel.from_pretrained(KOCLIP).to(DEVICE).eval(),
                            CLIPProcessor.from_pretrained(KOCLIP))
    return _cache["koclip"]


# ---------- 텍스트 탭 ----------

def analyze_text(text):
    text = (text or "").strip()
    if not text:
        return "문장을 입력하세요.", "", ""

    # 1) 스팸 분류 — BERT(3종 세트) + RULE + 하이브리드 (지시 2의 선택지 메뉴)
    from rule_spam import classify as rule_classify, score as rule_score
    tok, model, meta = get_spam()
    enc = tok([text], truncation=True, max_length=128, padding=True,
              return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    bert = int(probs.argmax())
    rule = rule_classify(text)
    name = lambda v: "🚨 스팸" if v else "✅ 정상"
    cls_md = (
        "### 📊 스팸 분류 — 4종 판정\n\n"
        f"| 방법 | 판정 | 근거 |\n|---|---|---|\n"
        # 성능 표기는 meta.json 연동 — 모델(가중치)을 교체하면 UI도 자동 갱신 (현재 가중치의 test셋 실측치)
        f"| BERT (test셋 정확도 {meta['metrics']['테스트정확도']}) | {name(bert)} | 확신 {probs[bert]:.1%} |\n"
        f"| RULE (0.12초·설명가능) | {name(rule)} | 걸린 신호 {rule_score(text)}개 |\n"
        f"| HYBRID-AND (오탐 0 지향) | {name(rule and bert)} | 만장일치제 |\n"
        f"| HYBRID-OR (미탐 최소 지향) | {name(rule or bert)} | 한 표제 |")

    # 2) PII 마스킹 (ko-pii — 룰+사전+체크섬, 학습 없음)
    # <PERSON_1> 류 토큰이 Markdown에서 HTML 태그로 해석돼 사라지므로 이스케이프 필수
    esc = lambda s: str(s).replace("<", "&lt;").replace(">", "&gt;")
    r = get_pii().process(text)
    pii_md = f"### 🛡️ PII 탐지·마스킹 (개인정보 관점)\n\n**마스킹**: {esc(r.text)}\n\n"
    if r.detections:
        # 민감도 = 유형에 붙는 등급(항목 속성). 토큰 없음 = 확신이 애매해 자동 마스킹을 보류한 항목(REVIEW)
        pii_md += "| 토큰(조치) | 유형 | 원문 | 민감도 | 확신 |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {esc(i.token) if i.token else '확신 애매 → 사람 검토'} | {i.detection.label} "
            f"| {esc(i.detection.text)} | {i.detection.risk_level.name} "
            f"| {i.detection.confidence:.0%} |" for i in r.detections)
        s = r.summary
        rationale = " · ".join(s.get("combined_rationale", []))
        ids = ", ".join(s.get("distinct_identifiers", [])) or "없음"
        quasi = ", ".join(s.get("distinct_quasi_identifiers", [])) or "없음"
        pii_md += (f"\n\n**PII 종합 위험도: {s.get('combined_risk')}** — {rationale}\n\n"
                   f"식별자(단독으로 개인 특정): **{ids}** · 준식별자(결합될수록 재식별 우려↑): **{quasi}**\n\n"
                   f"*민감도 = 그 유형이 얼마나 민감한 부류인가(항목 속성) · "
                   f"PII 종합 위험도 = 이 문장의 **개인정보만으로** 개인이 특정되는가 "
                   f"(스팸 등 다른 위험과 무관한 PII 한정 판정)*")
    else:
        pii_md += "탐지된 개인정보 없음."

    # 3) NER 개체 추출 (ner_klue.pt — 토큰 분류 + 구간 병합)
    # NER은 개체 추출기이지 PII 탐지기가 아님 — 개체마다 PII 관련성을 함께 표시
    import predict_ner
    NER_PII = {"PS": "⚠️ PII 후보 (인명 = 준식별자)",
               "LC": "일반 (상세 주소로 결합 시 준식별 소지)",
               "OG": "일반 개체", "DT": "일반 개체", "TI": "일반 개체", "QT": "일반 개체"}
    ents = predict_ner.tag_sentence(text, *get_ner())
    ner_md = "### 🏷️ NER 개체 추출 (일반 개체 인식)\n\n"
    if ents:
        ner_md += ("| 개체 | 유형 | PII 관련성 |\n|---|---|---|\n" + "\n".join(
            f"| {t} | {NER_KO.get(ty, ty)}({ty}) | {NER_PII.get(ty, '일반 개체')} |" for t, ty in ents)
            + "\n\n*NER의 PII 판정은 참고용 — 정식 PII 판정은 왼쪽 PII 칸(ko-pii) 담당. "
            "인명은 두 모듈의 접점 (룰의 인명 미탐을 NER이 보완하는 하이브리드 지점)*")
    else:
        ner_md += "추출된 개체 없음."
    return cls_md, pii_md, ner_md


# ---------- 검수·라벨링 탭 — 순환도의 "분류 결과 →[검수]→ 데이터셋 편입" 화살표의 실체 ----------
# 모델 판정은 초안(의사 라벨)일 뿐, 사람이 확정하는 순간에만 정답 가족으로 승격된다.
# 저장 계약 = text,label CSV → train_text.py에 코드 수정 없이 그대로 입력 가능 (순환 폐쇄).

REVIEW_CSV = REFS / "rnd-dataset-artifacts/export/datasets-own/ko-spam-reviewed.csv"


def draft_label(text):
    text = (text or "").strip()
    if not text:
        return "문장을 입력하세요.", None
    from rule_spam import classify as rule_classify, score as rule_score
    tok, model, _ = get_spam()
    enc = tok([text], truncation=True, max_length=128, padding=True,
              return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    bert, rule = int(probs.argmax()), rule_classify(text)
    name = lambda v: "🚨 스팸" if v else "✅ 정상"
    md = (f"**모델 초안 (의사 라벨 — 아직 정답 아님)**\n\n"
          f"- BERT: {name(bert)} (확신 {probs[bert]:.1%})\n"
          f"- RULE: {name(rule)} (걸린 신호 {rule_score(text)}개)\n\n"
          + ("두 방법 **일치** — 쉬운 사례일 가능성 (표본 확인 수준으로 충분)"
             if bert == rule else
             "⚠️ 두 방법 **불일치 — 검수 가치가 가장 높은 경계 사례** (전수 검수 권장)")
          + "\n\n아래 버튼으로 사람이 확정하는 순간에만 정답이 됩니다.")
    return md, {"text": text, "bert": bert, "rule": rule, "conf": float(probs[bert])}


def save_label(state, human_label):
    if not state:
        return "먼저 '초안 생성'으로 문장을 분석하세요."
    import pandas as pd
    text = state["text"]
    if REVIEW_CSV.exists() and text in pd.read_csv(REVIEW_CSV)["text"].astype(str).tolist():
        return "이미 검수된 문장입니다 — 중복 저장 건너뜀."
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not REVIEW_CSV.exists()
    with REVIEW_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["text", "label", "bert_초안", "rule_초안", "bert_확신", "검수일"])
        w.writerow([text, human_label, state["bert"], state["rule"],
                    round(state["conf"], 4), datetime.now().strftime("%Y-%m-%d %H:%M")])
    df = pd.read_csv(REVIEW_CSV)
    n_spam = int((df["label"] == 1).sum())
    flipped = int((df["label"] != df["bert_초안"]).sum())
    return (f"✅ 저장: [{'스팸' if human_label else '정상'}] {text[:40]}\n\n"
            f"**누적 {len(df)}건** (정상 {len(df) - n_spam} / 스팸 {n_spam}) · "
            f"사람이 초안을 뒤집은 건: {flipped}건\n\n"
            f"저장 위치: `datasets-own/ko-spam-reviewed.csv` (git 추적 — 재생성 불가 자산) · "
            f"`text,label` 계약이라 `DATA=이 파일`로 train_text.py에 바로 학습 가능")


# ---------- 이미지 탭 ----------

def analyze_image(image, prompts_text):
    if image is None:
        return None, "이미지를 업로드하세요.", ""

    # 1) YOLO — 무엇이 어디에 (박스)
    res = get_yolo().predict(image, verbose=False, conf=0.25)[0]
    annotated = res.plot()[:, :, ::-1]          # BGR → RGB
    counts = {}
    for c in res.boxes.cls.tolist():
        n = res.names[int(c)]
        counts[n] = counts.get(n, 0) + 1
    yolo_md = ("**탐지 객체**: " + ", ".join(f"{k}×{v}" for k, v in counts.items())) \
        if counts else "탐지된 객체 없음 (신뢰도 0.25 기준)."

    # 2) KoCLIP — 전체 상황 (프롬프트 = 분류 체계, 편집 즉시 반영·재학습 없음)
    prompts = [p.strip() for p in (prompts_text or DEFAULT_PROMPTS).splitlines() if p.strip()]
    model, proc = get_koclip()
    inputs = proc(text=prompts, images=[image], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(-1)[0]
    order = probs.argsort(descending=True)
    clip_md = "| 상황 후보 | 확신 |\n|---|---|\n" + "\n".join(
        f"| {'**' if i == 0 else ''}{prompts[k]}{'**' if i == 0 else ''} | {probs[k]:.1%} |"
        for i, k in enumerate(order.tolist()))
    if probs[order[0]] < 0.5:
        clip_md += "\n\n⚠️ 1위 확신이 50% 미만 — 후보에 없는 상황일 수 있음 (캐치올 클래스 확인)"
    return annotated, yolo_md, clip_md


# ---------- UI ----------

with gr.Blocks(title="탐지 AI 통합 데모") as demo:
    gr.Markdown("# 탐지 AI 통합 데모\n완료된 R&D 모듈 통합 (지시 1·2·4 산출물 재사용 — 신규 학습 없음)")
    with gr.Tab("텍스트 분석"):
        t_in = gr.Textbox(label="분석할 문장", lines=3,
                          placeholder="예: 신청인 홍길동 (880101-1234568) 무료 쿠폰 당첨! 010-1234-5678로 연락주세요")
        t_btn = gr.Button("분석", variant="primary")
        with gr.Row():
            t_cls = gr.Markdown(label="스팸 분류")
            t_pii = gr.Markdown(label="PII 마스킹")
            t_ner = gr.Markdown(label="NER 개체")
        t_btn.click(analyze_text, t_in, [t_cls, t_pii, t_ner])
    with gr.Tab("검수·라벨링"):
        gr.Markdown("모델 판정은 **초안(의사 라벨)**, 정답은 사람의 확정으로만 — "
                    "순환도의 \"분류 결과 →[검수]→ 데이터셋 편입\" 화살표. 확정분은 학습 데이터 산출물이 됩니다.")
        r_in = gr.Textbox(label="검수할 문장", lines=2, placeholder="분류가 애매하거나 새로 수집된 문장을 입력")
        r_btn = gr.Button("초안 생성 (모델 제안 보기)", variant="secondary")
        r_draft = gr.Markdown()
        r_state = gr.State()
        with gr.Row():
            r_ham = gr.Button("✅ 정상으로 확정", variant="primary")
            r_spam = gr.Button("🚨 스팸으로 확정", variant="stop")
        r_result = gr.Markdown()
        r_btn.click(draft_label, r_in, [r_draft, r_state])
        r_ham.click(lambda s: save_label(s, 0), r_state, r_result)
        r_spam.click(lambda s: save_label(s, 1), r_state, r_result)
    with gr.Tab("이미지 분석"):
        with gr.Row():
            with gr.Column():
                i_in = gr.Image(label="분석할 이미지", type="pil")
                i_prompts = gr.Textbox(label="상황 후보 (한 줄 = 후보 1개 — 편집 즉시 반영, 재학습 없음)",
                                       value=DEFAULT_PROMPTS, lines=6)
                i_btn = gr.Button("분석", variant="primary")
            with gr.Column():
                i_out = gr.Image(label="YOLO 박스")
                i_yolo = gr.Markdown()
                i_clip = gr.Markdown()
        i_btn.click(analyze_image, [i_in, i_prompts], [i_out, i_yolo, i_clip])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
