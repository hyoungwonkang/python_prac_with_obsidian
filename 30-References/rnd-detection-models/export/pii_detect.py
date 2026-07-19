"""
PII 탐지·마스킹 데모 — ko-pii 라이브러리 통합 (학습 없음).

플랫폼의 "PII 비식별화" 계층 대응. NER/YOLO와 달리 파인튜닝 모델이 아니라,
ko-pii(룰 + 사전 + 체크섬 기반, MIT)를 그대로 통합해 한국어 문장에서 개인정보를
탐지·마스킹한다. 주민번호·사업자번호 등은 체크섬으로 검증하고, 되돌리기 가능한
Vault에 원본을 보관한다.

실행:
  python pii_detect.py
"""
from ko_pii import Anonymizer, ProcessingMode

SAMPLES = [
    "신청인 홍길동 (880101-1234568) 연락처 010-1234-5678 이메일 hong@example.com",
    "계좌 110-234-567890 으로 입금, 차량번호 12가3456, 사업자 220-81-62517",
    "김철수 고객님 주소는 서울시 강남구 테헤란로 123 입니다.",
]


def main():
    # STRICT 모드 + tokenize 전략: 탐지 항목을 <TYPE_n> 토큰으로 치환(되돌리기 가능)
    anonymizer = Anonymizer(mode=ProcessingMode.STRICT, strategy="tokenize")

    for text in SAMPLES:
        result = anonymizer.process(text)
        print("=" * 70)
        print("원문 :", text)
        print("마스킹:", result.text)
        print(f"탐지 {len(result.detections)}건:")
        for item in result.detections:
            det = item.detection
            token = item.token or f"({item.action.name})"   # 치환 안 된 항목은 action 표시
            print(f"   {token:<16} ← [{det.label}] '{det.text}' "
                  f"(risk={det.risk_level.name}, conf={det.confidence:.2f})")
        # 되돌리기(Vault reveal) 데모 — 첫 토큰 원복
        if result.detections:
            tok = result.detections[0].token
            print(f"   Vault reveal: {tok} → {result.vault.reveal(tok)}")
        print("종합 위험도:", result.summary.get("combined_risk"))

    print("=" * 70)
    print("※ ko-pii는 학습이 아니라 룰+사전+체크섬 기반 라이브러리 — 33개 PII 범주 지원.")


if __name__ == "__main__":
    main()
