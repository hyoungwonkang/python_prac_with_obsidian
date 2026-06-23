---
title: MLOps · MLflow 기초 학습
tags: [reference, mlops, mlflow, ml]
---

# MLOps · MLflow 학습 정리

> 개발자를 위한 눈높이 노트

이 문서는 MLOps·MLflow를 처음 접하는 개발자 관점에서, 개념과 작동 원리를 비유와 함께 정리한 학습 노트입니다. LLM 학습 트랙([[../10-Projects/llm-from-scratch]])과 병행하는 개념 학습 노트입니다.

> 정본 출처 = `~/dev/python_prac_with_obsidian/30-References/mlops-mlflow.md`

> ⚠️ **작성 중** — MLOps 학습이 아직 진행 중입니다. 아래는 [[bert_ocr_practice_plan|BERT·OCR 학습정리]]와 동일한 하우스 스타일로 잡은 **틀(템플릿)**이며, 내용은 학습이 끝나는 대로 섹션별로 채웁니다.

## 목차

_(섹션이 확정되면 번호 + 앵커 링크로 채운다. 예: 1. [개념 — 큰 그림](#1-...))_

---

<!--
■ 작성 틀 (BERT·OCR 학습정리 하우스 스타일 — 두 파일에서 확인)

1. 부제 callout `> 개발자를 위한 눈높이 노트` → 문서 한 문단 소개
2. `## 목차` — 번호 목록 + 섹션 앵커 링크
3. `---` 구분선으로 섹션 분리
4. 번호 매긴 `## N. 제목 — 한 줄 부연` 섹션
   - 첫 섹션은 보통 "… — 큰 그림"(high-level 개념)
   - 필요하면 `### ① / ② / (1) / (2)` 소제목으로 분기
5. 표는 `| 구분 | A | B |` + `| --- | --- | --- |` 형식
6. 코드/다이어그램은 ``` 펜스 블록 (python 또는 plain)
7. 강조 박스는 blockquote callout:
   - `> 한 문장으로:` / `> 한 줄로:` — 섹션 핵심 압축
   - `> 참고 —` — 보충·정정·주의
   - `> 핵심은 …` — 요점 강조
8. 마지막 `## 핵심 요약` — blockquote로 전체를 한 줄에 꿰고, 보충 한두 문단
9. 톤: 개발자 비유 적극 사용(for 루프, base 이미지, grep, 미들웨어 체인 등), 초보자 눈높이

■ 다룰 후보 주제 (학습하며 추가/조정)
- "높은 레벨(high-level)"의 의미 — 추상화 수준
- ML 수명 주기 상세 뷰 (Data Prep → Train & Tune → Deploy & Monitor → Inference)
- MLOps 4대 원칙 (버전 제어 · 자동화 · 지속적 X · 모델 거버넌스)
- MLflow의 역할 — 실험 추적 · 모델 레지스트리 · 아티팩트 관리
-->

## 학습 노트

_(아직 작성 전 — 학습 진행에 따라 위 틀대로 `## N. 섹션`을 채운다.)_

## 용어집

_(등장한 핵심 용어를 한 줄 정의로 누적: high-level, Artifact Lineage, Feature Store, reproducibility, IaC, CI/CD/CT/CM, model drift 등)_

## 핵심 요약

_(전체 내용이 모이면 blockquote 한 줄 꿰기로 마무리)_

## 참고 링크

- MLflow 공식 문서: https://mlflow.org/docs/latest/index.html

## 관련 노트

- [[../10-Projects/llm-from-scratch]] — 병행 중인 LLM 학습 트랙(5장 사전훈련·6장 분류 FT에 학습 추적 접점)
- [[bert_ocr_practice_plan]] — 같은 하우스 스타일의 학습정리(BERT·OCR)
- [[notion-mcp]] — Notion 미러 등록·동기 규칙
- [[_References]] — 레퍼런스 인덱스
