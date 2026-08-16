# 1P. 문제 정의

## 1. UI/UX 관찰

- 메인 영역에서 이벤트·프로모션·에디터 큐레이션 등 **비개인화 영역 비중이 큼**
- 비개인화 영역은 **업데이트 주기가 길고 편성이 고정적**
- 개인화 추천은 **가로 캐러셀 단위**로 제공되며 캐러셀마다 추천 목적이 다름

## 2. 목적

**구독 전환 및 유지를 높이는 방향으로 메인 가로 캐러셀 개인화 추천 시스템 설계**

## 3. 가설

### 가설 1. 노출 편향과 중복 추천이 개인화 추천의 소비 효율을 저하시킬 수 있다

- 비개인화 편성에서 반복적으로 노출된 도서가 **개인화 추천에서도 과도하게 추천될 수 있다.**
- 비개인화·개인화 영역에서 동일 도서가 중복 추천되면 **새로운 도서를 노출할 슬롯의 기회비용이 발생할 수 있다.**

### 가설 2. 단일 Ranker의 소비확률 최적화 효과는 캐러셀 특성에 따라 다를 수 있다

- 캐러셀마다 후보 생성 기준과 추천 목적이 다르다.
- 따라서 **동일한 Ranker Score와 실제 소비 간 관계가 캐러셀별로 다르게 나타날 수 있다.**
- 일부 캐러셀에서는 Ranker Score와 소비 간 관계가 약하거나 불안정할 수 있다.

---

# 2P. 가설 검정 및 추천 목표

## 1. 가설 1 — 노출 편향 및 중복 추천

### ① 비개인화 노출 편향이 개인화 추천으로 전이되는가

**검정법:** 다변량 Logistic Regression

```text
개인화 추천 여부
~ 과거 비개인화 노출량
+ 기존 도서 인기도
+ Item 특성
+ User 특성
```

- 통제 후에도 비개인화 노출량의 계수가 유의하게 양수  
  → **비개인화 노출 편향의 개인화 추천 전이 존재**

### ② 중복 추천의 증분가치가 충분한가

**검정법:** Offline 로그 비교 또는 A/B Test + Lift 비교

```text
중복 허용 : 비개인화 A + 개인화 A
vs
중복 제거 : 비개인화 A + 개인화 B
```

- **사용자 전체 소비지표** 비교
- 중복 제거 시 소비가 유의하게 높음  
  → **중복 추천의 슬롯 기회비용 존재**

## 2. 가설 2 — 캐러셀별 Ranker 효과 차이

### ① 캐러셀별 Ranker Score–소비 관계 비교

```text
소비지표
~ Ranker Score
+ Carousel
+ Ranker Score × Carousel
+ Controls
```

캐러셀별 비교:

- Ranker Score–소비 관계의 **기울기 β**
- β의 **유의성 / 신뢰구간**
- 캐러셀 간 β 차이
- `Ranker Score × Carousel` 상호작용 유의성

**판단**

- 캐러셀별 β 차이가 유의함  
  → **Ranker Score의 소비 예측 효과가 캐러셀별로 다름**
- 차이가 유의하지 않음  
  → **가설 미지지 → 공통 Ranker 유지**

> 현재 로그 분석은 Ranker Score와 소비 간 캐러셀별 **연관성 차이**를 검정하며, Ranker 적용 자체의 인과적 증분효과를 의미하지는 않음.

### ② 효과가 약한 캐러셀의 임계값 분석

Ranker Score–소비 관계가 특히 약한 캐러셀을 식별.

예: **「내 서재에 담은 책과 비슷한 책」**

```text
소비지표
~ f(Item Similarity)
+ Ranker Score
+ 인기도
+ 노출 위치
+ User / Item 특성
```

- Item Similarity와 소비지표의 반응곡선 추정
- 소비 성과가 악화되기 시작하는 **Similarity Threshold τ** 도출
- 해당 캐러셀 Re-ranking 시 **`Similarity ≥ τ` 제약 적용**

## 3. 검정 결과별 설계 판단

| 검정 결과 | 설계 |
|---|---|
| 노출 편향 전이 존재 + 중복 가치 낮음 | **편향 완화 + 중복 제거** |
| 노출 편향 전이 존재 + 중복 가치 높음 | 중복 유지, 편향 완화만 검토 |
| 노출 편향 없음 + 중복 가치 낮음 | **중복 제거만 적용** |
| 노출 편향 없음 + 중복 가치 높음 | 기존 정책 유지 |
| 캐러셀별 Ranker 효과 차이 존재 | **효과가 약한 캐러셀 분석 → 캐러셀별 제약 적용** |
| 캐러셀별 Ranker 효과 차이 없음 | **공통 Ranker 유지** |

## 4. 추천 목표

| 추천 목표 | 기대 효과 | 측정 지표 |
|---|---|---|
| **노출 편향·중복 완화** | 특정 도서 집중 완화 및 추천 슬롯 효율 향상 | **열람 작품 수 / 열독시간 / 완독 수** |
| **캐러셀별 Ranking 최적화** | Ranker 효과가 약한 캐러셀의 소비 성과 개선 | **캐러셀별 열람률 / 열독시간 / 완독률** |

---

# 3P. 캐러셀 기획 및 추천 모델 구조

## 1. Candidate Generation

| 캐러셀 | Candidate 기준 | 역할 |
|---|---|---|
| **나에게 딱 맞는 책** | **iALS / EASE^R** | 사용자 행동 기반 개인화 후보 생성 |
| **내 서재에 담은 책과 비슷한 책** | **Text Embedding + ANN** | 서재에 담은 도서와 의미적으로 유사한 후보 생성 |

## 2. 전체 추천 구조

```text
Candidate Generation
        ↓
      Ranker
        ↓
Carousel-specific Constraint
        ↓
 Duplicate Filter
        ↓
Temporal xQuAD Re-ranking
        ↓
 ε-greedy Exploration
        ↓
Final Recommendation
```

## 3. 구조 판단

| 구조 판단 | 선택 이유 |
|---|---|
| **Multi-stage** | 검증된 추천 구조로 메인 영역 안정성 확보 + 단계별 실험 용이 |
| **Hybrid Candidate** | 캐러셀 목적에 따라 CF·Content 기반 후보 생성 방식 선택 |
| **Ranking / Re-ranking 분리** | 소비확률 최적화와 캐러셀 제약·편향 완화 목표를 분리해 최적화 |
| **캐러셀별 Constraint** | Ranker 효과가 약한 캐러셀에 데이터 기반 고유 제약 적용 |
| **중복 제거 분리** | 비개인화 영역과의 중복을 슬롯 효율 제약으로 명시적 처리 |

## 4. Ranker

**Shallow → GBDT → Deep 단계적 검증 [1]**

| Arm | 모형 | 주요 입력 |
|---|---|---|
| **A — Shallow** | Logistic / Linear | User · Item · Context |
| **B — GBDT** | LightGBM | 동일 |
| **C — Deep** | DNN | User · Item Representation · Context |

복잡한 추천 모델이 잘 튜닝된 단순 baseline을 항상 능가하지 않는다는 재현 연구를 근거로 **모델 복잡도별 증분 성능 비교**.

## 5. 캐러셀별 Constraint 및 노출 정책

| 방법 | 활용 데이터 | 역할 |
|---|---|---|
| **Carousel-specific Constraint** | 캐러셀별 목적 신호 | Ranker 효과가 약한 캐러셀에 데이터 기반 최소 조건 적용 |
| **Duplicate Filter** | 현재 비개인화 영역 노출 목록 | 중복 추천으로 인한 슬롯 기회비용 방지 |
| **Temporal xQuAD** | Ranker Score + 과거 노출 이력 | 과노출 도서 집중 완화 및 long-tail 노출 보완 [2] |
| **ε-greedy Exploration** | 추천 가능 후보군 | 저노출 도서 노출 + OPE·향후 정책 학습용 로그 확보 [3] |
* Exploration은 캐러셀 특성에 따라 선택 적용. 유사도 등 캐러셀 고유 기준이 중요한 경우 무작위 노출이 추천 품질을 훼손할 수 있으므로, 미적용하거나 해당 Constraint를 만족하는 후보군 내부에서만 수행.
### 각주

**[1]** Ferrari Dacrema et al. (2019); Rendle et al. (2020)  
**[2]** Abdollahpouri & Burke (2019), *Reducing Popularity Bias in Recommendation Over Time*  
**[3]** Swaminathan & Joachims (2015), *Counterfactual Risk Minimization*; Wang et al. (2017), *Off-policy Evaluation in Contextual Bandits*

---

# 4P. 시스템 아키텍처 및 운영 설계

## 1. 전체 시스템 구조

```text
Web / App
    ↓
Backend API / BFF
    ↓
┌──────────── Recommendation Service ────────────┐
│                                               │
│  추천 파이프라인 제어 계층                    │
│  (데이터 조회 · 단계 실행 순서 · 결과 조합)   │
│                    ↓                          │
│         Recommendation Pipeline               │
│                                               │
└───────────┬────────────────────┬──────────────┘
            ↓                    ↑
   User / Catalog Provider   Versioned Model Artifact
                                  ↑
                           Offline Training
                                  ↑
                       Behavior Log / Feature
```

## 2. 핵심 설계 판단

| 설계 판단 | 선택 이유 | 구체적인 구조 반영 |
|---|---|---|
| **Recommendation Service 분리** | 추천 변경·장애를 기존 Backend와 격리 | `Backend/BFF → 내부 HTTP → Recommendation Service` |
| **Serving / Learning 분리** | 학습 작업이 Online 응답 속도와 안정성에 영향을 주는 것 방지 | `Offline Training → Versioned Artifact → Serving 시 Load` |
| **추천 Pipeline 모듈화** | Ranker·Constraint·Re-ranking 정책을 독립적으로 실험·교체 | 추천 파이프라인 제어 계층이 각 컴포넌트를 순차 호출 |
| **Dependency Injection** | 실험 시 Pipeline 전체 코드 변경 방지 | `container.py`에서 각 구현체 주입 |
| **데이터 접근 추상화** | 실제 DB·Feature Store 기술과 추천 로직 분리 | `CatalogRepository / UserRepository / PolicyProvider` Interface |
| **Online 연산 최소화** | 전체 카탈로그 대상 실시간 연산으로 인한 응답 지연 방지 | 사전 계산 CF + ANN Index 기반 Top-K 조회 |
| **Batch Inference** | 후보별 개별 추론 호출 비용 감소 | 후보 Feature를 행렬화해 Ranker에서 일괄 추론 |
| **Version 관리** | Ranker·Constraint·Policy 실험 재현 및 Rollback | Model / Feature / Policy Version 기록 |

## 3. 학습 및 배포 구조

```text
Behavior Log + Point-in-time Feature
              ↓
        Time-based Split
              ↓
        Offline Training
              ↓
          Evaluation
              ↓
    Versioned Model Artifact
              ↓
Recommendation Service 시작 시 Load
```

| 대상 | 업데이트 정책 | 이유 |
|---|---|---|
| **CF Candidate** | 매일 | 최신 소비 이력 반영 |
| **Ranker** | 주 1회 + 데이터량 조건 | Online 학습 없이 검증된 Artifact만 배포 |
| **Content Embedding / ANN Index** | 신규·수정 도서 발생 시 | 전체 재생성 대신 증분 반영 |
| **캐러셀 Constraint** | 분석·실험 결과 갱신 시 | 캐러셀별 임계값을 Ranker와 독립적으로 관리 |
| **조기 재학습** | Drift 발생 시 | 정기 주기 사이의 데이터·성능 변화 대응 |

---

# 5P. 성능 평가 및 실험 설계

## 1. 평가 원칙

| 판단 | 근거 | 평가 방향 |
|---|---|---|
| **Offline은 후보 선별** | Offline 평가는 Online 성과의 불완전한 대리변수이며 구조적 한계가 존재 [1][3] | 열위 Candidate / Ranker 사전 제거 |
| **OPE는 보조 평가** | 기존 로그에는 이전 추천 정책·UI의 노출 편향이 포함됨 [2] | support가 확보된 로그에서 IPS / SNIPS / DR |
| **Online이 최종 판단** | 추천 정책 변경의 실제 소비 효과는 실사용 환경에서 검증 필요 [2] | 추천 목표별 A/B Test |
| **장기 Business KPI는 참고** | 유료전환·Retention은 관측 주기가 길고 가격·프로모션·콘텐츠 등 외부 영향이 큼 | 추천 직접 성과지표에서 제외, 장기 추적 |
| **복잡도–운영비 Trade-off** | 추가 성능이 모델 복잡도·Latency·Compute 증가를 정당화해야 함 | Online Lift 대비 Serving / Training Cost 비교 |

## 2. 평가 및 승격 흐름

```text
G0. Offline Evaluation
Candidate : Recall@K · Coverage
Ranker    : NDCG@K
Constraint: Threshold / Holdout 검증
        +
제한적 OPE : IPS · SNIPS · DR
        ↓
G1. Online A/B Test
추천 목표별 소비지표 개선
        ↓
G2. 운영성 검증
Latency · Serving / Training Cost
        ↓
최종 승격
```

- **G0:** 열위 Candidate / Ranker 제거 및 캐러셀별 Constraint의 Holdout 검증
- **OPE:** ε-greedy로 propensity와 support가 확보된 범위에서 정책 효과 추정
- **G1:** 실제 사용자 소비 개선 여부 확인
- **G2:** 개선폭이 추가 운영비·Latency를 정당화하는지 판단

## 3. 추천 목표별 Online 평가

| 추천 목표 | 실험 | 성과지표 |
|---|---|---|
| **노출 편향·중복 완화** | 기존 정책 vs Duplicate Filter + Temporal xQuAD | **사용자 전체 열람 작품 수 / 열독시간 / 완독 수** |
| **캐러셀별 Ranking 최적화** | Ranker-only vs Ranker + Carousel Constraint | **해당 캐러셀 열람률 / 열독시간 / 완독률** |

장기 **유료 전환율·Retention**은 관측 기간이 길고 추천 외 요인의 영향이 커 직접 승격 KPI로 사용하지 않는다.

## 4. 단계별 실험

| 실험 | 검증 목적 |
|---|---|
| **Ranker** | Shallow → GBDT → Deep의 증분 성능 검증 |
| **Carousel Constraint** | 데이터에서 도출한 캐러셀별 임계값의 소비 성과 개선 검증 |
| **Duplicate Filter** | 중복 제거에 따른 슬롯 효율 및 전체 소비 개선 검증 |
| **Temporal xQuAD** | 노출 편향 완화에 따른 전체 소비 개선 검증 |
| **ε-greedy Exploration** | 소비 성능과 Exploration 데이터 확보 간 Trade-off 검증 |

**Ranker → Carousel Constraint → Duplicate Filter → Re-ranking → Exploration 순으로 개별 효과 검증**

## 5. 모델 운영비용 추정

밀리의서재 공개 규모 기준:

- 누적 회원 **1,000만 명**
- 콘텐츠 약 **24만 권**
- 정확한 MAU / QPS는 비공개
- 과제 가정: **MAU 100만 / 월 추천 요청 2,000만~5,000만**
- 평균 약 **8~19 QPS**, Peak 약 **40~200 QPS**

| Ranker | 예상 Serving 구성 | 월 Compute 추정* | 판단 |
|---|---|---:|---|
| **Shallow** | CPU 4vCPU × 2~4 replicas | **약 $300~600** | 가장 저렴 |
| **LightGBM** | CPU 8vCPU × 4~8 replicas | **약 $1,100~2,300** | 현실적인 중간안 |
| **DNN** | 고사양 CPU 또는 GPU Serving | **약 $2,000+** | 유의미한 Online Lift 필요 |

캐러셀별 Constraint·Duplicate Filter·Temporal xQuAD는 후보 축소 이후의 후처리이므로, **운영비 차이의 핵심은 Candidate Retrieval과 Ranker 복잡도**로 본다.

**승격 기준:**  
Online 소비지표 개선폭이 추가 **Serving / Training Cost + p95 Latency 증가**를 정당화할 때만 복잡한 모델로 승격.

\\* 추천 모델 Compute만의 추정치이며 DB, 로그 저장, 네트워크, 모니터링 비용은 제외.

### 각주

**[1]** Hidasi & Czapp (2023), *Widespread Flaws in Offline Evaluation of Recommender Systems*  
**[2]** Gruson et al. (2019), *Offline Evaluation to Make Decisions About Playlist Recommendation Algorithms*  
**[3]** Castells & Moffat (2022), *Offline Recommender System Evaluation: Challenges and New Directions*
