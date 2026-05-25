# Vercel 프로젝트 전환 절차

vault 안의 한 학습 프로젝트(`10-Projects/todo-app`)를 Vercel에 배포한 뒤, 다음 학습 프로젝트(`10-Projects/blog-app` 등)로 **같은 Vercel 프로젝트를 덮어쓰는** 절차.

## 배경

- 이 vault는 통째로 GitHub 레포(`python_prac_with_obsidian`)에 올림.
- Vercel은 그 안의 한 폴더만 **Root Directory**로 지정해서 배포.
- 새 학습 프로젝트로 넘어갈 때, 같은 Vercel 프로젝트의 Root Directory만 바꾸면 됨.
- 이렇게 하면: ① 포트폴리오용 URL 1개만 유지, ② 무료 플랜 한도 절약, ③ 과거 학습 프로젝트는 vault·git에 그대로 보존.

대안 (병렬 운영)은 [[#대안-병렬-운영]] 참고.

## 전환 체크리스트

새 프로젝트가 `10-Projects/blog-app`이라고 가정.

```bash
# 1. 새 프로젝트 폴더에 Vercel 필수 파일 준비
#    - vercel.json
#    - api/index.py  (from app.main import app)
#    - requirements.txt
#    - .vercelignore (.venv, *.db, .env 등)

# 2. 변경사항 push
cd /home/user1/python_prac_with_obsidian
git add 10-Projects/blog-app/
git commit -m "feat(blog-app): Vercel 배포 설정 추가"
git push origin main

# 3. Vercel 프로젝트의 Root Directory + 이름 변경 (한 명령)
vercel api -X PATCH "/v9/projects/todo-app" \
  -F rootDirectory="10-Projects/blog-app" \
  -F name="blog-app"

# 4. 환경변수 갱신 (필요한 경우)
vercel env rm DATABASE_URL production
vercel env add DATABASE_URL production
# → 프롬프트에 새 프로젝트용 DB URL 입력

# 5. 재배포 트리거
vercel --prod
# 또는 main에 빈 커밋 push로 자동 배포 유도
```

## 핵심 명령 설명

### `vercel api -X PATCH "/v9/projects/<현재이름>"`

Vercel REST API의 프로젝트 업데이트 엔드포인트를 CLI가 인증해서 호출.

- `-X PATCH` — HTTP 메서드
- `/v9/projects/<이름>` — 대상 프로젝트 (이름 또는 ID로 식별)
- `-F key=value` — 폼 필드 (CLI가 JSON으로 변환해서 전송)

수정 가능한 필드 예시:
- `rootDirectory` — 빌드 시작 위치 (레포 루트 기준 상대 경로, 슬래시 없음)
- `name` — 프로젝트 이름 (URL alias에도 반영됨)
- `framework` — 프리셋 (`nextjs`, `other` 등)
- `buildCommand`, `installCommand`, `outputDirectory`

### `vercel env`

- `vercel env ls` — 현재 등록된 환경변수 목록
- `vercel env add <KEY> <환경>` — 환경변수 추가 (production / preview / development)
- `vercel env rm <KEY> <환경>` — 환경변수 삭제
- `vercel env pull .env.local` — 클라우드 → 로컬 파일로 받기

## 자주 빠뜨리는 것

| 항목 | 증상 |
|---|---|
| `rootDirectory` 앞에 `/` 붙임 | "Path not found" 빌드 실패 |
| 새 프로젝트에 `vercel.json` 없음 | 기본 정적 사이트로 인식 → 404 |
| `requirements.txt` 새 폴더에 없음 | Python 의존성 미설치 → import 에러 |
| 환경변수 미갱신 | 새 프로젝트가 옛 DB에 연결됨 |
| `api/index.py`에서 import 경로 잘못 | 함수 시작 시 ImportError로 500 |

## 대안 (병렬 운영)

여러 학습 프로젝트를 **동시에 살려두고** 싶다면:

- 같은 GitHub 레포를 Vercel에 **다시 import**하면서 Root Directory만 다르게 지정.
- 한 레포 → 여러 Vercel 프로젝트 가능.
- 단점: Vercel 프로젝트가 누적되어 관리 부담 증가.

## 자동화

전환 빈도가 잦아지면 `scripts/vercel-switch.sh`로 묶기:

```bash
#!/usr/bin/env bash
# 사용법: ./scripts/vercel-switch.sh blog-app
set -e
NEW="$1"
NEW_ROOT="10-Projects/$NEW"
[ -d "$NEW_ROOT" ] || { echo "Error: $NEW_ROOT not found"; exit 1; }

CURRENT=$(vercel project ls 2>/dev/null | awk 'NR>3 {print $1; exit}')

vercel api -X PATCH "/v9/projects/$CURRENT" \
  -F rootDirectory="$NEW_ROOT" \
  -F name="$NEW"

echo "✓ Switched: $CURRENT → $NEW"
echo "  Next: update env vars, then vercel --prod"
```

GitHub Actions로 옮기면 로컬 환경 없이도 트리거 가능하지만, 학습용엔 과한 인프라.

## 관련 노트

- [[_References]] (인덱스로 돌아가기)
- [[../10-Projects/todo-app]] — 현재 배포 중인 첫 프로젝트
