"""
통합 UXUI 데모 — 지시 3. 완료된 R&D 모듈 5개를 탭 2개짜리 데모 UI로 통합 (제작, 신규 학습 없음).

  텍스트 탭: BERT 스팸 분류(지시 1 산출물 3종 세트) + RULE·하이브리드(지시 2)
             + PII 마스킹(ko-pii) + NER 개체 추출(ner_klue.pt)
  이미지 탭: YOLO 박스(yolov8n) + KoCLIP 한국어 상황 태그(지시 4 — 프롬프트 실시간 편집)

실행:
  ~/rnd-env/bin/python app.py    # → http://127.0.0.1:7860
모델은 첫 사용 시 로딩(지연 로딩) — 탭별 첫 응답만 수 초 걸림.
"""
import json
import os
import sys
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
DEFAULT_PROMPTS = "온라인 도박 사이트 화면\n일상 생활 사진\n음식 사진\n거리와 차량 사진\n동물 사진\n문서나 서류"

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
    pii_md = f"**마스킹**: {esc(r.text)}\n\n"
    if r.detections:
        pii_md += "| 토큰 | 유형 | 원문 | 위험도 |\n|---|---|---|---|\n" + "\n".join(
            f"| {esc(i.token) if i.token else '-'} | {i.detection.label} | {esc(i.detection.text)} "
            f"| {i.detection.risk_level.name} |" for i in r.detections)
        pii_md += f"\n\n종합 위험도: **{r.summary.get('combined_risk')}**"
    else:
        pii_md += "탐지된 개인정보 없음."

    # 3) NER 개체 추출 (ner_klue.pt — 토큰 분류 + 구간 병합)
    import predict_ner
    ents = predict_ner.tag_sentence(text, *get_ner())
    ner_md = ("| 개체 | 유형 |\n|---|---|\n" + "\n".join(
        f"| {t} | {NER_KO.get(ty, ty)}({ty}) |" for t, ty in ents)) if ents \
        else "추출된 개체 없음."
    return cls_md, pii_md, ner_md


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
        clip_md += "\n\n⚠️ 1위 확신이 50% 미만 — 후보에 없는 상황일 수 있음 (받이 클래스 확인)"
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
