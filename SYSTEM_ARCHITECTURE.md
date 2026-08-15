# 개인화 캐러셀 추천 시스템 아키텍처

## 1. 설계 범위

새 `README.md.md`의 두 목표를 실제 서비스로 제공하는 구조를 설명한다.

1. 비개인화 영역에서 만들어진 노출 편향과 중복 추천을 완화한다.
2. 가입 시 선택한 관심 분야로 신규 회원 Cold-start를 개선한다.

추천 API, 후보 생성, Ranker A/B/C, Duplicate Filter, Temporal xQuAD, ε-greedy와 학습 파이프라인을 구현 범위로 둔다. 인증, 회원·도서 DB, 행동 이벤트 수집 서버와 메시지 브로커는 기존 백엔드/데이터 시스템이 제공한다고 가정한다.

## 2. 전체 시스템 구조

```text
Web / App
    │ 메인 화면 요청
    ▼
Backend API / BFF
    │ 내부 HTTP: 사용자·캐러셀·비개인화 노출 목록
    ▼
┌──────────────── Recommendation Service ────────────────┐
│ FastAPI                                                │
│   └─ Orchestrator                                      │
│       ├─ Candidate Generator                           │
│       │   ├─ 개인화: CF + Content                     │
│       │   └─ Cold-start: Category + Popularity         │
│       ├─ Ranker: A0 / A / B / C                       │
│       ├─ Duplicate Filter                              │
│       ├─ Temporal xQuAD                                │
│       └─ ε-greedy Exploration                          │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
       User/Catalog Provider       Versioned Model Artifact

Behavior Log + Feature Snapshot
               │
               ▼
       Offline Training Job
       A / B / C + Evaluation
               │
               └──────────────▶ Versioned Model Artifact
```

프론트엔드는 추천 서비스에 직접 접근하지 않는다. 백엔드가 인증과 화면 구성을 담당하고 추천 결과의 작품 ID를 카탈로그 정보와 결합한다. 추천 장애 시 백엔드가 기본 캐러셀을 반환할 수 있다는 장점도 있다.

WebSocket 대신 내부 HTTP를 사용한다. 캐러셀은 요청 하나에 결과 하나를 반환하는 구조라 지속적인 양방향 연결이 필요하지 않고 timeout, 재시도와 API 문서화가 단순하기 때문이다.

## 3. 구현 컴포넌트

| 컴포넌트 | 책임 | 코드 |
| --- | --- | --- |
| Recommendation API | 요청 검증과 결과·버전 반환 | `api/`, `main.py` |
| Orchestrator | 추천 단계 실행 순서 제어 | `core/orchestrator.py` |
| Candidate Generator | 캐러셀별 후보 생성 | `core/candidates.py` |
| Ranker | 공통 feature를 A0/A/B/C로 점수화 | `core/ranking.py`, `core/ml_rankers.py` |
| Duplicate Filter | 현재 비개인화 영역과 같은 작품 제거 | `core/eligibility.py` |
| Temporal xQuAD | 관련성과 시간 감쇠된 노출 공정성 결합 | `core/reranking.py` |
| ε-greedy | 낮은 확률로 후보를 무작위 노출하고 propensity 기록 | `core/slate.py` |
| Training | 시간 분할, A/B/C 학습과 평가 | `training/` |
| Repository Interface | 실제 저장 기술과 추천 로직 분리 | `ports.py` |

각 단계는 Orchestrator가 생성하지 않고 `container.py`에서 주입한다. Ranker만 바꾸고 Candidate·Filter·xQuAD를 고정하거나, Ranker를 고정한 채 xQuAD만 실험하기 위한 구조다.

## 4. 온라인 요청 흐름

```http
POST /v1/recommendations/carousels
X-Request-Id: r_123
```

```json
{
  "user_id": "u_1",
  "session_id": "s_1",
  "carousel_type": "FOR_YOU",
  "limit": 10,
  "non_personalized_work_ids": ["w_10", "w_20"],
  "context": {"device": "ANDROID"}
}
```

처리 순서는 다음과 같다.

1. User/Catalog Provider에서 profile과 추천 가능한 도서를 읽는다.
2. `FOR_YOU`는 장기 이용 이력과 콘텐츠 feature를, `INTEREST_COLD_START`는 onboarding 관심 분야와 분야 내 인기도를 사용한다.
3. Ranker가 후보의 관련성 점수를 계산한다.
4. 비개인화 영역에서 이미 노출된 작품을 제거한다.
5. Temporal xQuAD가 반복 노출이 적은 도서를 보완한다.
6. ε 확률로 한 슬롯을 탐색 후보로 교체하고 선택 확률을 기록한다.
7. 작품×포맷 ID, 위치, 추천 사유와 model/feature/policy version을 반환한다.

`non_personalized_work_ids`를 요청에 포함한 이유는 추천 서비스가 현재 화면 전체를 직접 조회하지 않게 하기 위해서다. 화면을 구성하는 백엔드가 이미 알고 있는 정보를 전달하므로 별도 DB 조회가 필요 없다.

## 5. 데이터 계약과 자유 가정

데이터는 과제에서 자유롭게 가정할 수 있으므로 다음 schema가 존재한다고 본다.

| 데이터 | 필드 예시 | 사용 |
| --- | --- | --- |
| User | user_id, age, onboarding_genres, 장기 genre affinity | 후보·Ranker |
| Item | work_id, format_id, genre, author, popularity, text embedding, 판권 | 후보·필터 |
| Exposure | user_id, work_id, 영역, 위치, timestamp, policy, propensity | xQuAD·OPE |
| Outcome | impression 이후 일정 기간 내 consumed 여부 | Ranker label |

README에서 소비 신호 가설이 유의하지 않은 것으로 가정했으므로 click/start/dwell/complete를 별도 multi-task head로 예측하지 않는다. 대신 노출 후 정의된 기간 안에 유효 소비가 발생했는지를 `consumed` label로 둔다. 실제 환경에서는 열람 시작 또는 최소 독서 시간을 조합해 이 label을 정할 수 있다.

## 6. Ranker 학습과 배포

세 Arm은 같은 후보, feature와 `consumed` label을 사용한다.

| Arm | 구현 | 비교 목적 |
| --- | --- | --- |
| A | Logistic Regression | 해석 가능한 학습 기준선 |
| B | LightGBM | 비선형 feature interaction의 추가 가치 |
| C | DNN + user/item embedding | representation learning의 추가 가치 |

```text
Behavior Log + Point-in-time Feature
                 ↓
           Time-based Split
                 ↓
            A / B / C 학습
                 ↓
       Offline Evaluation / Calibration
                 ↓
       Versioned Model Artifact
                 ↓
    Recommendation Service 시작 시 로딩
```

Serving과 Learning은 분리한다. 추천 API는 요청 중 학습하지 않고 시작 시 선택한 artifact를 한 번 로딩한다. 학습 실패가 온라인 latency에 영향을 주지 않고 모델 버전별 비교와 rollback이 가능하다.

Arm B/C는 데이터가 없어서 제외한 것이 아니라 가정한 schema로 완전히 구현한다. 실행 시 각각 LightGBM과 PyTorch 의존성이 필요하며, 실제 성능 판단만 실제 로그 또는 생성한 실험 데이터로 수행한다.

### 초기 재학습 주기

| 대상 | 기본 주기 | 추가 조건 |
| --- | --- | --- |
| CF·인기도 후보 | 매일 | 신규 도서 대량 유입 시 조기 실행 |
| Ranker A/B/C | 주 1회 | 신규 label 1만 건 이상일 때만 실행 |
| Ranker 조기 학습 | 정기 주기 전 | Feature PSI ≥ 0.2 또는 NDCG 하락 ≥ 0.03 |
| 콘텐츠 embedding | 신규·수정 도서 유입 시 | 증분 index 반영 |

`training/schedule.py`의 `RetrainingPolicy`가 이 판단을 코드로 제공한다. 실제 시간 기반 실행은 Airflow 등 배치 오케스트레이터가 담당한다고 가정한다. 고정 주기만 사용하지 않고 데이터량과 drift를 함께 보는 이유는 변화가 없는데 불필요하게 학습하거나 급격한 변화에 일주일간 대응하지 못하는 일을 줄이기 위해서다. 임계값은 초기 가정이며 운영 데이터로 조정한다.

## 7. Temporal xQuAD와 Exploration

Temporal xQuAD는 Ranker 관련성과 사용자의 과거 노출 이력을 결합한다.

```text
repeated_exposure = Σ exp(-0.05 × 노출 후 경과일)
exposure_fairness = exp(-repeated_exposure)
final_score = (1-λ) × rank_score + λ × exposure_fairness
```

최근 반복 노출된 책은 fairness가 낮고 오래전에 노출된 책은 다시 회복된다. `λ`는 편향 완화 강도이며 Ranker와 분리해 실험한다.

ε-greedy는 추천 가능한 도서 중 한 슬롯을 확률적으로 선택한다. `propensity = ε / feasible_item_count`를 응답에 포함해 OPE 로그로 사용할 수 있게 한다. 결정론적인 xQuAD만으로는 action probability가 생기지 않으므로 별도 exploration을 둔다.

## 8. 장애와 fallback

| 상황 | 처리 |
| --- | --- |
| 신규회원 관심 분야 없음 | 적격 인기 도서 fallback |
| 개인화 이력 없음 | 콘텐츠·인기도 점수 사용 |
| 일부 후보 부족 | 적격 catalog에서 인기 후보 보충 |
| 모델 artifact 로딩 실패 | 배포 실패 또는 A0를 명시적으로 선택 |
| 추천 API timeout | 백엔드 기본 캐러셀 반환 |

자동 fallback으로 서로 다른 Arm 결과를 섞으면 실험 해석이 어려우므로 실제 사용된 Arm과 버전을 반드시 응답·로그에 남긴다.

### 응답 지연 budget

내부 Recommendation API의 초기 목표는 `p95 150ms`, 서비스 deadline은 `250ms`로 둔다.

| 단계 | 목표 budget |
| --- | ---: |
| User/Catalog provider | 50ms |
| Candidate retrieval | 30ms |
| Ranker batch inference | 40ms |
| Duplicate/xQuAD/Exploration | 10ms |
| 직렬화와 여유 | 20ms |

후보 생성은 온라인에서 24만 권 전체를 순회하지 않고 repository가 CF, 카테고리, exploration 상위 K를 조회하는 계약으로 구성했다. 실제 구현은 사전 계산된 CF 결과, ANN index와 카테고리별 cache로 이 계약을 만족해야 한다. Arm B/C는 후보를 행렬로 묶어 한 번에 추론한다.

Orchestrator는 `feature_ms`, `candidate_ms`, `ranker_ms`, `rerank_ms`, `elapsed_ms`를 기록한다. 각 단계 뒤에 deadline을 확인하며 250ms를 넘으면 API가 504를 반환하고 백엔드가 기본 캐러셀을 사용한다. 외부 provider와의 실제 HTTP 호출에도 남은 deadline보다 짧은 client timeout을 별도로 적용해야 한다.

## 9. 실서비스 교체 지점

현재 인메모리 User/Catalog adapter는 API를 독립 실행하기 위한 예제다. 실서비스에서는 `CatalogRepository`, `UserRepository`, `PolicyProvider` 구현만 내부 API나 feature store client로 교체한다. DB schema, 이벤트 ingestion과 인증을 추천 소스에 포함하지 않은 것은 다른 팀 영역을 다시 구현하지 않고 시스템 경계를 분명하게 하기 위해서다.
