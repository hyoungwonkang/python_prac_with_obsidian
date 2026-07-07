# md 보고서 변환 파이프라인 — mermaid 도식 포함 md → 단일 파일 전달

> 2026-07-07 확립 (전이학습 보고서 6호에서 첫 사용). "md의 mermaid를 이미지로 바꾸면
> 경로 지정+별첨밖에 없나?"라는 문제의 해법 — **파일 하나로 도식까지 전달하는 3경로.**

## 결론 요약

| 경로 | 결과물 | 언제 |
|---|---|---|
| **PDF** (권장) | 도식 박힌 pdf 1개 | 상사 보고·결재·인쇄 — 어디서나 열림 |
| **자립형 HTML** | SVG를 파일 안에 내장한 html 1개 (외부 참조 0) | 브라우저로 볼 사람에게 |
| **노션 붙여넣기** | 변환 불필요 | 채널이 노션이면 mermaid 코드블록을 자체 렌더링 — 최단 경로 |
| ~~base64 임베딩 md~~ | md 1개 | 비추천 — GitHub·노션 등이 data URI 차단 |

## 파이프라인 (검증된 명령)

전제: node(npx)·Chrome. 첫 실행만 mermaid-cli가 크로미움 다운로드(수 분), 이후 수 초.

```bash
# ① mermaid 블록 추출 → SVG 렌더링
#    (md에서 ```mermaid``` 블록을 .mmd로 추출한 뒤)
npx -y @mermaid-js/mermaid-cli -i d0.mmd -o d0.svg -b white

# ② md → HTML 본문 (mermaid 블록은 placeholder로 치환해 두고)
npx -y marked --gfm -i report.md -o body.html

# ③ HTML 템플릿 래핑: <head>에 한글 폰트·표 CSS, placeholder 자리에 SVG 원문 삽입
#    → 외부 참조 없는 자립형 html (검증: grep -c "<svg" / 외부 src 0건)

# ④ PDF 출력
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --no-pdf-header-footer --print-to-pdf="report.pdf" "file://$PWD/report.html"

# ⑤ 시각 검증 (1페이지 래스터화 → 눈으로 확인)
sips -s format png -Z 900 report.pdf --out check.png
```

## 요령·함정

- **SVG를 `<img src>`가 아니라 본문에 원문 삽입** — 그래야 단일 파일(경로 의존 0).
- CSS에 `@media print { h2 { page-break-after: avoid } }` — PDF 페이지 나눔 품질.
- 한글 폰트: `'Apple SD Gothic Neo','Malgun Gothic'` 지정 (맥/윈도우 겸용).
- 검증 2종: HTML은 `grep "<svg"`+외부 참조 0건, PDF는 `sips`로 1페이지 이미지화해 눈 확인.
- 데스크톱 폴더는 셸 권한(TCC)에 가끔 막힘 — 파일 존재는 grep/cp 성공 로그로 확인 가능.

## 관련

- 첫 적용: [[rnd-detection-models/06-전이학습보고서]] (html 25KB·pdf 523KB 산출)
- mermaid 원본 문서: [[rnd-detection-models/05-도식도]]
