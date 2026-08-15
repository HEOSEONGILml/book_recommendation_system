# 1P. 문제 정의

## 1. UI/UX 관찰

- 메인 영역에서 이벤트·프로모션·에디터 큐레이션 등 **비개인화 영역 비중이 큼**
- 비개인화 영역은 **업데이트 주기가 길고 편성이 고정적**
- 개인화 추천은 **가로 캐러셀 단위**로 제공

## 2. 목적

**구독 전환 및 유지를 높이는 방향으로 메인 가로 캐러셀 개인화 추천 시스템 설계**

## 3. 가설

### 1) 개인화–탐색 / 노출 편향

- 비개인화 편성의 노출 편향이 특정 도서의 중복 추천을 만들 수 있음
- 이를 완화하며 개인화–탐색 균형을 조정하면 소비지표가 개선될 것

### 2) Cold-start

- 초기 무료회원의 추천 경험은 유료 전환과 연관
- 가입 시 선호정보 기반 Cold-start 추천이 유효할 것

### 3) 소비 신호

- 구독 모델에서는 클릭보다 열람·열독시간·완독 등의 소비 신호가 중요할 것

### 4) 장기 취향–최근 관심

- 장기 취향과 최근 행동을 함께 반영하는 것이 중요할 것

---

# 2P. 가설 검정 및 추천 목표

> **소비 신호 / 장기 취향–최근 관심 가설은 검정 결과 유의하지 않은 것으로 가정하여 이후 설계에서 제외**

## 1. 노출 편향 검정

### ① 노출 편향이 개인화 추천으로 전이되는가

**검정법:** 다변량 Logistic Regression

```text
개인화 추천 여부
~ 과거 비개인화 노출량
+ 인기도
+ Item 특성
+ User 특성
```

- 비개인화 노출량 효과가 유의하면 → **추천 편향 전이 존재**

### ② 중복 추천의 증분가치가 충분한가

**검정법:** Offline 로그 비교 또는 A/B Test + Lift 비교

```text
중복 허용: 비개인화 A + 개인화 A
vs
중복 제거: 비개인화 A + 개인화 B
```

- **사용자 전체 소비지표** 비교
- 중복 제거 시 소비가 높으면 → **중복 슬롯의 기회비용 존재**

## 2. Cold-start 검정

**검정법:** Offline 신규회원 Cohort 비교 또는 A/B Test + Lift 비교

```text
기존 기본 추천
vs
가입 시 선호 기반 추천
```

- 신규회원 초기 소비 비교
- 선호 기반 추천 효과가 유의하면 → **Cold-start 전략에 반영**

## 3. 검정 결과별 설계 판단

### 노출 편향 × 중복 가치

| 노출 편향 | 중복 가치 | 설계 |
|---|---|---|
| 존재 | 낮음 | **편향 완화 + 중복 제거** |
| 존재 | 높음 | 중복 유지, 편향 대응 별도 검토 |
| 없음 | 낮음 | **중복 제거만 적용** |
| 없음 | 높음 | 기존 정책 유지 |

### Cold-start

| 결과 | 설계 |
|---|---|
| 효과 있음 | **선호 기반 Cold-start 추천 적용** |
| 효과 없음 | 기존 / 인기·Content 기반 추천 |

## 4. 추천 목표

| 추천 목표 | 기대 효과 | 측정 지표 |
|---|---|---|
| **편향 완화** | 중복 추천 감소, 슬롯 효율 향상 | 열람 작품 수 / 열독시간 / 완독 수 |
| **Cold-start 개선** | 신규회원 초기 소비 활성화 | 초기 열람 작품 수 / 열독시간 / 이용일수 |

**측정**

- 편향 완화: A/B Test 기준 **사용자 전체 소비**
- Cold-start: A/B Test 기준 **신규회원 초기 소비**

---

# 3P. 캐러셀 기획 및 추천 모델 구조

## 1. 캐러셀별 Candidate Generation

| 캐러셀 | 추천 목표 연계 | 활용 데이터 | Candidate Generation |
|---|---|---|---|
| **나에게 딱 맞는 책** | 기본 개인화 | 열람·열독·완독 이력 + 도서 콘텐츠 | **iALS / EASE^R + Text Embedding·ANN** |
| **내 관심분야의 책** | Cold-start 개선 | 가입 시 관심 분야·선호 정보 | **Category Matching + 카테고리 내 Popularity** |

## 2. 전체 추천 구조

```text
Catalog / User Data
        ↓
[Candidate Generation]
 ├─ 개인화 후보
 └─ Cold-start 후보
        ↓
[Ranker]
        ↓
[Duplicate Filter]
        ↓
[Temporal xQuAD Re-ranking]
        ↓
[ε-greedy Exploration]
        ↓
Final Recommendation
```

| 구조 판단 | 선택 이유 |
|---|---|
| **Multi-stage** | 검증된 추천 구조로 메인 영역 안정성 확보 + 단계별 실험 용이 |
| **Hybrid Candidate** | 개인화·Cold-start 목적별 후보 생성 방식 조합 |
| **Ranking / Re-ranking 분리** | 사용자 적합도와 편향 완화 목표를 분리해 최적화 |
| **중복 제거 분리** | 중복 노출을 슬롯 효율 제약으로 명시적 처리 |

## 3. Ranker

**Shallow → GBDT → Deep 단계적 검증 [1]**

| Arm | 모형 | 주요 입력 |
|---|---|---|
| **A — Shallow** | Logistic / Linear | User · Item · Context |
| **B — GBDT** | LightGBM | 동일 |
| **C — Deep** | DNN | User · Item Representation · Context |

복잡한 추천 모델이 잘 튜닝된 단순 baseline을 항상 능가하지 않는다는 재현 연구를 근거로 모델 복잡도별 성능 비교.

## 4. 편향 완화 및 Exploration

| 방법론 | 활용 데이터 | 근거 |
|---|---|---|
| **Duplicate Filter** | 현재 비개인화 영역 노출 목록 | 중복 추천으로 인한 슬롯 기회비용 방지 |
| **Temporal xQuAD** | Ranker Score + **과거 노출 이력** | 과노출 도서 집중 완화 및 long-tail 노출 보완 [2] |
| **ε-greedy Exploration** | 추천 가능 후보군 | 저노출 도서의 노출 기회 + **OPE·향후 정책 학습용 로그 확보** [3] |

### 각주

**[1]** Ferrari Dacrema et al. (2019); Rendle et al. (2020)  
**[2]** Abdollahpouri & Burke (2019), *Reducing Popularity Bias in Recommendation Over Time* — Temporal xQuAD  
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
| **Serving / Learning 분리** | 학습 작업이 Online 응답 속도와 안정성에 영향 주는 것 방지 | `Offline Training → Versioned Artifact → Serving 시 Load` |
| **추천 Pipeline 모듈화** | 특정 단계만 교체하여 독립적인 실험 가능 | 추천 파이프라인 제어 계층이 독립 컴포넌트를 순차 호출 |
| **Dependency Injection** | 실험 시 Pipeline 전체 코드 변경 방지 | `container.py`에서 각 구현체 주입 |
| **데이터 접근 추상화** | 실제 DB·Feature Store 기술과 추천 로직 분리 | `CatalogRepository / UserRepository / PolicyProvider` Interface |
| **Online 연산 최소화** | 전체 카탈로그 대상 실시간 연산으로 인한 응답 지연 방지 | 사전 계산 CF + ANN Index + Category Cache 기반 Top-K 조회 |
| **Batch Inference** | 후보별 개별 추론 호출 비용 감소 | 후보 Feature를 행렬화해 Ranker에서 일괄 추론 |
| **Version 관리** | 실험 재현·모델 비교·Rollback 가능 | Versioned Model Artifact + 실제 사용 Version 로그 |

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
| **CF / Popularity** | 매일 | 최근 소비·인기도 변화 반영 |
| **Ranker** | 주 1회 + 데이터량 조건 | Online 학습 없이 검증된 Artifact만 배포 |
| **Content Embedding** | 신규·수정 도서 발생 시 | 전체 재생성 대신 증분 Index 반영 |
| **조기 재학습** | Drift 발생 시 | 정기 주기 사이의 급격한 데이터 변화 대응 |

---

# 5P. 성능 평가 및 실험 설계

## 1. 평가 원칙

| 판단 | 근거 | 평가 방향 |
|---|---|---|
| **Offline은 후보 선별** | Offline 평가는 Online 성과의 불완전한 대리변수이며 평가 자체에도 구조적 한계가 존재 [1][3] | 열위 Candidate / Ranker 사전 제거 |
| **OPE는 보조 평가** | 기존 로그에는 이전 추천 정책·UI에 의한 노출 편향이 포함됨 [2] | support가 확보된 로그에서 IPS / SNIPS / DR |
| **Online이 최종 판단** | 실제 추천 정책 변화의 소비 효과는 실사용 환경에서 검증 필요 [2] | 추천 목표별 A/B Test |
| **장기 Business KPI는 참고** | 유료전환·Retention은 관측 주기가 길고 가격·프로모션·콘텐츠 등 다양한 요인의 영향을 받음 | 추천의 직접 성과지표에서 제외, 장기 추적 |
| **복잡도–운영비 Trade-off** | 추가 성능이 모델 복잡도·Latency·Compute 증가를 정당화해야 함 | Online Lift 대비 Serving / Training Cost 비교 |

## 2. 평가 및 승격 흐름

```text
G0. Offline Evaluation
Candidate : Recall@K · Coverage
Ranker    : NDCG@K
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

- **G0:** 명백히 열위한 Candidate / Ranker 제거
- **OPE:** ε-greedy Exploration으로 propensity와 support가 확보된 범위에서 정책 효과 추정
- **G1:** 추천 목표별 실제 소비지표 개선 여부 확인
- **G2:** 성능 개선이 추가 운영비와 Latency를 정당화하는지 판단

## 3. 추천 목표별 Online 평가

| 추천 목표 | 실험 | 성과지표 |
|---|---|---|
| **편향 완화** | 기존 정책 vs Duplicate Filter + Temporal xQuAD | **사용자 전체 열람 작품 수 / 열독시간 / 완독 수** |
| **Cold-start 개선** | 기존 신규회원 추천 vs 선호 기반 Candidate | **신규회원 초기 열람 작품 수 / 열독시간 / 이용일수** |

장기 **유료 전환율·Retention**은 추천 외 요인의 영향이 크므로 직접 승격 KPI로 사용하지 않는다.

## 4. 단계별 실험

| 실험 | 검증 목적 |
|---|---|
| **Ranker** | Shallow → GBDT → Deep의 증분 성능 검증 |
| **Cold-start Candidate** | 가입 시 선호 기반 추천의 신규회원 초기 소비 개선 검증 |
| **Temporal xQuAD** | 노출 편향 완화에 따른 사용자 전체 소비 개선 검증 |
| **ε-greedy Exploration** | 소비 성능과 Exploration 데이터 확보 간 Trade-off 검증 |

**Ranker → Cold-start / Re-ranking → Exploration 순으로 개별 효과 검증**

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

24만 권 수준에서는 Candidate Generation보다 **Ranker 복잡도 증가에 따른 Serving 비용 차이**를 운영비 판단의 핵심으로 본다.

**승격 기준:**  
Online 소비지표 개선폭이 추가 **Serving / Training Cost + p95 Latency 증가**를 정당화할 때만 복잡한 모델로 승격.

\* 추천 모델 Compute만의 추정치이며 DB, 로그 저장, 네트워크, 모니터링 비용은 제외.

### 각주

**[1]** Hidasi & Czapp (2023), *Widespread Flaws in Offline Evaluation of Recommender Systems*  
**[2]** Gruson et al. (2019), *Offline Evaluation to Make Decisions About Playlist Recommendation Algorithms*  
**[3]** Castells & Moffat (2022), *Offline Recommender System Evaluation: Challenges and New Directions*
