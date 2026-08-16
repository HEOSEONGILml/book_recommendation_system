# 구현 안내

## 1. 구현 기준

`README.md`의 설계를 실행 가능한 형태로 구현했다. 캐러셀별 후보와 제약을 독립 모듈로 두고 다음 순서로 처리한다.

```text
캐러셀별 Candidate
→ Ranker A0/A/B/C
→ Carousel-specific Constraint
→ 비개인화 영역 Duplicate Filter
→ Temporal xQuAD
→ ε-greedy Exploration
→ FastAPI 응답
```

데이터가 주어지지 않은 부분은 구현 가능한 schema와 sample catalog를 가정했다. Arm B/C도 제외하지 않고 학습, artifact 저장과 API 추론 adapter까지 포함한다.

## 2. 실행

```powershell
uv sync --extra dev
uv run uvicorn recommendation_service.main:app --reload
```

- API 문서: `http://127.0.0.1:8000/docs`
- 추천: `POST /v1/recommendations/carousels`
- 상태 확인: `GET /health/live`, `GET /health/ready`

```powershell
uv run pytest
uv run ruff check .
```

## 3. 코드 구성과 이유

| 코드 | 구현 | 선택 이유 |
| --- | --- | --- |
| `core/candidates.py` | CF 개인화와 서재 도서 기반 콘텐츠 후보 | 캐러셀 목적에 맞는 retrieval 사용 |
| `core/ranking.py` | A0와 Arm A 추론, 공통 feature | 단순 baseline과 ML 모델 비교 |
| `core/ml_rankers.py` | Arm B/C 추론 | 동일 Ranker 계약으로 교체 가능 |
| `core/constraints.py` | 유사 도서 캐러셀의 similarity 임계값 | Ranker가 약한 캐러셀의 최소 조건 보장 |
| `core/eligibility.py` | 판권·연령과 비개인화 중복 제거 | 점수와 무관한 제약을 명시적으로 처리 |
| `core/reranking.py` | Temporal xQuAD | 반복 노출 편향을 Ranker와 분리해 조절 |
| `core/slate.py` | ε-greedy와 propensity | 저노출 도서 및 OPE용 로그 확보 |
| `training/` | A/B/C 학습·평가·artifact | 모델 복잡도의 증분 가치 비교 |
| `api/` | FastAPI 요청·응답 | 백엔드가 호출 가능한 최소 serving 경계 |

## 4. API 예시

```powershell
$body = @{
  user_id = "u_1"
  session_id = "s_1"
  carousel_type = "LIBRARY_SIMILAR"
  limit = 10
  non_personalized_work_ids = @("w_001", "w_010")
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/recommendations/carousels `
  -Headers @{"X-Request-Id" = "r_local_1"} `
  -ContentType application/json `
  -Body $body
```

`FOR_YOU`는 CF 방식의 개인화 후보를, `LIBRARY_SIMILAR`는 서재에 담긴 작품 embedding과 가까운 콘텐츠 후보를 사용한다. 로컬 sample user는 `w_005`, `w_011`을 서재 작품으로 가정한다.

## 5. 모델 학습

CSV schema는 다음과 같이 가정한다.

```text
occurred_at,user_id,work_id,carousel_type,
bias,source_score,genre_affinity,item_similarity,popularity,consumed
```

`consumed`는 노출 후 정해진 관측 기간 안에 유효 소비가 발생했는지를 나타내는 이진 label이다.

```powershell
uv run millie-train --arm A --input data/train.csv --output artifacts --version ranker-a-001
uv run millie-train --arm B --input data/train.csv --output artifacts --version ranker-b-001
uv run millie-train --arm C --input data/train.csv --output artifacts --version ranker-c-001
```

Arm B/C는 다음 의존성을 추가로 사용한다.

```powershell
uv sync --extra dev --extra ml
```

학습된 모델을 API에 연결한다.

```powershell
$env:MILLIE_RANKER_ARM = "B"
$env:MILLIE_MODEL_MANIFEST = "artifacts/ranker-b-001/manifest.json"
uv run uvicorn recommendation_service.main:app
```

Arm A는 직접 구현한 Logistic Regression, Arm B는 LightGBM classifier, Arm C는 user/item embedding을 포함한 DNN이다. 동일 feature와 시간 분할을 사용해 구조 차이의 효과를 비교한다.

### 재학습 판단

`training/schedule.py`는 다음 초기 정책을 구현한다.

- Candidate 모델: 매일 갱신
- Ranker: 주 1회, 신규 label 1만 건 이상일 때 실행
- Feature PSI 0.2 이상 또는 NDCG 0.03 이상 하락 시 Ranker 조기 학습

스케줄 실행기는 과제 범위에 포함하지 않고, 배치 오케스트레이터가 `RetrainingPolicy.decide()` 결과를 읽어 `millie-train`을 실행한다고 가정한다. 주기와 임계값은 환경 데이터로 조정할 운영 파라미터다.

## 6. 정책 설정

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `MILLIE_RANKER_ARM` | `A0` | 사용할 Ranker |
| `MILLIE_MODEL_MANIFEST` | 없음 | 학습 artifact 위치 |
| `MILLIE_XQUAD_LAMBDA` | `0.2` | 관련성 대비 노출 편향 완화 강도 |
| `MILLIE_EPSILON` | `0.05` | 탐색 슬롯을 사용하는 확률 |
| `MILLIE_REQUEST_TIMEOUT_MS` | `250` | 추천 서비스 내부 deadline |
| `MILLIE_SIMILARITY_THRESHOLD` | `0.55` | `LIBRARY_SIMILAR` 최소 cosine similarity |

λ, ε과 similarity threshold는 코드의 정답값이 아니라 분석·A/B Test 대상이다. 응답에 model/feature/policy version과 exploration propensity를 내려 실험 로그와 연결한다.

## 7. 응답 지연 고려

- Candidate repository는 전체 catalog 대신 최대 100개의 top-K 후보를 반환한다.
- LightGBM과 DNN은 후보를 batch로 추론한다.
- 모델 artifact는 요청마다 읽지 않고 프로세스 시작 시 한 번 로딩한다.
- 응답 diagnostics에 단계별 latency와 전체 latency를 포함한다.
- 250ms를 넘으면 504를 반환하고 백엔드가 기본 캐러셀로 대체한다.

초기 성능 목표는 내부 API p95 150ms다. 실제 provider adapter에는 connection/read timeout을 설정하고, 부하 테스트에서 A/B/C별 p95·p99를 측정해야 한다.

## 8. 구현하지 않은 부분

- 회원·도서 DB와 인증
- 실제 CF 학습 batch와 ANN index 구축 인프라
- 행동 이벤트 수집 서버와 메시지 큐
- 사용자별 실험군 배정 플랫폼
- 배포·autoscaling과 비용 집계 시스템

이 항목은 데이터가 없어서 포기한 기능이 아니라 다른 시스템과 연결되는 부분이다. 추천 코드에는 Repository와 API 계약을 남겨 교체 가능하게 했다. 과제에서 평가할 핵심인 Candidate, A/B/C Ranker, 편향 완화, Exploration과 평가 코드는 모두 구현 범위에 포함한다.
