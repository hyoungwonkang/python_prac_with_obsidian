---
title: 지시 마스킹(instruction masking) 현재 트렌드
tags: [reference, llm, finetuning, research-trend]
---

# 지시 마스킹(instruction masking) 현재 트렌드

> 교재(Raschka 2025) 연습문제 7.2 "지시와 입력 토큰을 -100으로 마스킹"의 배경 리서치.
> **2026-07 웹 조사 시점 기준** — 이후 갱신 시 날짜 추가할 것.
> 학습 맥락: [[../10-Projects/llm-from-scratch/llm-ch7-instruct]] §연습문제 7.2.

## 한 줄 결론

**마스킹(-100)은 여전히 프레임워크 기본값이자 표준 관행이지만, 연구 흐름은
"마스킹 O/X 이분법" → "지시문 loss를 얼마나 반영할지의 가중치 다이얼"로 이동.**
데이터가 작고 응답이 짧으면 마스킹을 끄는(또는 약하게 켜는) 쪽이 나을 수 있다 — 정답은 실험으로.

## 시간순 흐름

| 시기 | 내용 |
|---|---|
| ~2024 (표준) | 지시문 -100 마스킹, 응답만 채점이 지배적 관행. HF 등 주요 프레임워크 기본값 (지금도) |
| 2024 PLW | "Does Prompt Loss Matter?" — 마스킹을 0/1이 아닌 **가중치(prompt loss weight)**로 일반화. **응답 짧은 데이터**: 지시문 loss를 0.01~0.5로 **조금 반영**하는 게 더 좋음. **응답 긴 데이터**: 이 설정 자체가 결과에 거의 무영향 |
| 2024~25 IM | "Instruction Modeling"(Loss Over Instructions) — **데이터 적고(<1.5만) 지시문 길고 응답 짧으면 마스킹 안 하는 쪽이 우세** (전 구간 채점이 과적합 완화). 교재 연습문제가 인용하는 계열. Raschka 본인도 매거진에서 재검증: "비마스킹이 이긴다, 단 응답 길이 비율·데이터 크기에 달림" |
| 2025 WIT | TACL "On the Effect of Instruction Tuning Loss on Generalization" — 5모델×3데이터셋×5벤치마크. **완전 마스킹(0)도 완전 포함(1)도 최적 아님**. "지시문 낮은 가중치 + 응답 중간~높은 가중치"가 일관 최고. 이후 선호 정렬(DPO 등)의 출발점으로도 우수 |
| 2026 상반기 | 구간 단위를 넘어 **토큰 단위 선별**로 진화 — ProFit(확률 기반으로 학습할 토큰 선택) 등 |

## 연습문제 7.2에 적용하면

- rasbt 데이터셋 = **1,100건 + 짧은 응답** → 정확히 "마스킹 안 하는 게 나을 수 있다"고 지목된 조건.
- 따라서 과제의 올바른 자세 = "마스킹 구현"이 아니라 **마스킹/비마스킹 두 run을 MLflow로 비교**
  (val loss + 생성 품질). 교재 기본기와 최신 경향 확인을 한 번에.

## Sources (2026-07 조사)

- [Raschka — LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
- [On the Effect of Instruction Tuning Loss on Generalization (WIT, TACL 2025, arXiv:2507.07817)](https://arxiv.org/abs/2507.07817)
- [Instruction Fine-Tuning: Does Prompt Loss Matter? (PLW, arXiv:2401.13586)](https://arxiv.org/abs/2401.13586)
- [ProFit: Probability-Guided Token Selection (arXiv:2601.09195)](https://arxiv.org/pdf/2601.09195)

## 관련 노트

- [[../10-Projects/llm-from-scratch/llm-ch7-instruct]] — 7장 정본 노트 (마스킹 구현 방법 포함)
- [[llm-glossary]] — LLM 개념 용어집
