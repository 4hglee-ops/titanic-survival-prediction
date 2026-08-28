# Decision Log

이 문서는 Titanic 생존 예측 실험에서 **무엇을 선택했는지보다 왜 그렇게 선택했는지**를 기록합니다.

과거 학습 당시의 판단과 GitHub 정리 과정에서 다시 검토한 판단을 구분합니다. 당시 기록에서 명시적으로 확인되지 않는 이유는 사후에 그럴듯하게 만들어 넣지 않습니다.

---

## D01. Kaggle test를 검증 데이터로 사용하지 않는다

### 질문
Kaggle에서 제공하는 `test.csv`를 모델 성능 확인에 사용할 수 있는가?

### 판단
사용하지 않는다. Kaggle test에는 `Survived` 정답이 없기 때문에 로컬 성능 검증용 데이터가 아니다.

### 선택
- `train.csv` 내부에서 개발 데이터와 hold-out을 분리한다.
- 모델·피처·threshold 선택은 개발 데이터에서만 수행한다.
- hold-out은 선택이 끝난 뒤 마지막에 한 번 평가한다.
- Kaggle test는 최종 `submission.csv` 생성에만 사용한다.

### 의미
Kaggle 제출 데이터와 로컬 검증 데이터를 역할별로 분리해, 제출용 test를 평가 데이터처럼 해석하는 오류를 피한다.

---

## D02. Accuracy 하나만으로 모델을 선택하지 않는다

### 질문
Titanic 이진 분류에서 어떤 지표를 중심으로 모델을 비교할 것인가?

### 판단
Accuracy와 함께 Precision, Recall, F1, ROC-AUC를 확인한다. 실험 후보 비교와 threshold 탐색에서는 F1을 주요 기준으로 사용한다.

### 이유
Accuracy만 보면 한 클래스에 유리한 예측이 가려질 수 있다. F1은 Precision과 Recall의 균형을 확인할 수 있고, ROC-AUC는 특정 threshold에 고정되지 않은 분리 성능을 보조적으로 볼 수 있다.

### 현재 구현
동일한 `StratifiedKFold` 5-Fold 조건에서 Accuracy, Precision, Recall, F1, ROC-AUC를 함께 계산한다.

---

## D03. 먼저 여러 모델을 같은 조건에서 비교한다

### 질문
처음부터 한 모델만 튜닝할 것인가?

### 선택
다음 모델을 동일한 5-Fold 교차검증 조건에서 비교했다.

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

### 과거 학습에서 확인한 점
기본 피처와 맥락 피처 단계에서는 HGB와 XGB가 강한 성능을 보였다. 따라서 Random Forest가 처음부터 명백한 최종 모델이었던 것은 아니다.

### 의미
최종 RF는 단순 선호로 정한 것이 아니라 이후 튜닝과 threshold 비교까지 이어진 실험 후보 중 하나였다.

---

## D04. 승객 개인 정보에서 가족·사회적 맥락을 파생한다

### 질문
`Pclass`, `Sex`, `Age`, `Fare` 같은 원본 변수만으로 충분한가?

### 가설
Titanic에서는 승객 개인 속성뿐 아니라 **누구와 함께 탑승했는지, 어떤 사회적 역할을 가졌는지**가 생존과 관련될 수 있다.

### 실험한 주요 피처
- `FamilySize = SibSp + Parch + 1`
- `IsAlone`
- 이름에서 추출한 `Title`
- `CabinKnown`
- `Surname`
- `TicketGroupSize`
- `FamilySurvival`
- `FarePerPerson`
- `AgeGroup`
- `Mother`

### 최종 판단
모든 파생 피처를 무조건 유지하지 않았다. 반복 실험 결과가 좋았던 `FamilySize`, `IsAlone`, `Title`을 핵심 맥락 피처로 유지하고, 이후 Ticket 기반 피처를 추가 검토했다.

---

## D05. `CabinKnown`은 아이디어가 타당해 보여도 제외했다

### 질문
Cabin의 결측 여부 자체가 생존 정보를 담고 있지 않을까?

### 가설
객실 정보가 기록된 승객과 그렇지 않은 승객 사이에 객실 등급·사회경제적 조건 차이가 있을 수 있으므로 `CabinKnown`을 추가하면 성능이 좋아질 수 있다.

### 결과
과거 학습 기록에서 HGB + `FamilySize` + `IsAlone` + `Title`의 F1은 약 `0.7733`이었다. 여기에 `CabinKnown`을 추가했을 때 F1이 약 `0.7662`로 하락했다.

### 판단
`CabinKnown`은 최종 피처에서 제외했다.

### 의미
도메인상 그럴듯한 피처라도 실제 검증 성능이 나빠지면 유지하지 않는다.

---

## D06. Random Forest는 튜닝 후 다시 비교한다

### 질문
기본 모델 비교에서 HGB/XGB가 강했는데 왜 RF를 최종 후보로 남겼는가?

### 과거 학습 과정
Random Forest에 대해 `RandomizedSearchCV`로 과적합 제어 관련 파라미터를 탐색했다. 이후 RF, HGB, XGB 각각의 OOF probability에서 F1 기준 threshold를 다시 탐색했다.

당시 기록에서 확인되는 결과는 다음과 같다.

| 모델 | 선택 threshold | F1 |
|---|---:|---:|
| Tuned RF | `0.46` | **0.7852** |
| HGB | `0.53` | `0.7814` |
| XGB | `0.42` | `0.7761` |

RF의 threshold `0.46` 결과는 Precision 약 `0.7559`, Recall 약 `0.8169`, F1 약 `0.7852`였다.

### 판단
당시 실험에서는 **RF + Title + FamilySize + IsAlone, threshold 0.46**을 F1 기준 최종 후보로 선택했다.

### 주의
당시 기록에서 “왜 다른 모델보다 먼저 RF를 튜닝 대상으로 정했는가”라는 명시적인 이유까지는 확인되지 않았다. 따라서 해석 가능성이나 속도 같은 이유를 사후에 만들어 넣지 않는다.

---

## D07. 기본 threshold `0.5`를 고정값으로 보지 않는다

### 질문
분류 probability가 0.5 이상이면 무조건 생존로 분류하는 것이 최적인가?

### 판단
아니다. threshold는 모델 자체와 별개의 의사결정 변수다.

### 선택
개발 데이터의 OOF probability만 사용해 threshold를 탐색하고 F1이 가장 높은 값을 선택한다.

### 이유
학습 데이터에 직접 fit한 probability로 threshold를 고르면 과적합된 기준이 될 수 있다. OOF prediction을 사용하면 각 샘플의 probability가 해당 샘플을 학습하지 않은 모델에서 생성된다.

---

# GitHub 재검증 단계

아래 결정은 과거 학습 당시 결과를 그대로 옮긴 것이 아니라, **2026-08-27~28에 프로젝트를 GitHub용으로 다시 정리하면서 검증 구조를 재점검한 결과**다.

---

## D08. 전처리와 Ticket 통계를 Pipeline 내부로 이동한다

### 질문
교차검증 전에 전체 개발 데이터에서 전처리나 Ticket 통계를 계산해도 되는가?

### 판단
Fold별 학습 정보만 사용하도록 Pipeline 내부에서 계산한다.

### 선택
- 결측치 처리
- 스케일링
- one-hot encoding
- `TicketGroupSize`

을 학습 Fold 기준으로 계산한다.

검증 Fold에서 처음 등장한 Ticket은 `TicketGroupSize = 1`로 처리한다.

### 의미
Fold 밖의 정보가 전처리나 그룹 통계에 섞이는 데이터 누수를 줄인다.

---

## D09. Ticket 동행 정보를 다시 실험한다

### 가설
같은 Ticket을 사용하는 승객은 함께 여행했을 가능성이 있고, Ticket별 요금은 여러 승객의 합산 요금일 수 있다.

### 추가 피처
- `TicketGroupSize`
- `FarePerPerson = Fare / TicketGroupSize`

### 결과
현재 재구성된 실험에서 튜닝 RF + 맥락 피처 + Ticket 피처의 OOF threshold 최적값은 `0.47`이었다.

- Accuracy: `0.834`
- Precision: `0.766`
- Recall: `0.817`
- F1: **`0.791`**

### 판단
현재 수행한 실험 중 가장 높은 OOF F1을 기록해 최종 후보로 사용했다.

### 표현상의 주의
Ticket 피처를 적용한 HGB/XGB를 동일 조건으로 다시 비교한 것은 아니다. 따라서 “모든 가능한 모델 중 최적”이라고 표현하지 않고 **“현재 수행한 실험 중 가장 높은 결과”**라고 기록한다.

---

## D10. hold-out은 마지막에 한 번만 확인한다

### 선택 완료 후 결과
- Accuracy: `0.816`
- Precision: `0.750`
- Recall: `0.783`
- F1: `0.766`
- ROC-AUC: `0.853`

### 판단
OOF F1 `0.791`보다 hold-out F1이 낮아졌지만, hold-out 성능에 맞춰 다시 모델이나 threshold를 조정하지 않았다.

### 이유
hold-out 결과를 보고 다시 선택을 바꾸기 시작하면 hold-out도 사실상 개발 데이터가 된다. 이 실험에서는 최종 선택 이후의 일반화 확인이라는 역할을 유지한다.

---

## D11. Ticket 피처의 한계를 숨기지 않는다

현재 교차검증은 승객 단위 `StratifiedKFold`다. 같은 Ticket을 가진 승객이 서로 다른 Fold에 들어갈 수 있다.

`TicketFeatureBuilder`는 학습 Fold에서 Ticket별 그룹 크기를 계산하므로 직접적인 전체 데이터 누수는 막았지만, 같은 Ticket 그룹 구성원이 train/validation 양쪽에 존재할 수 있다는 구조적 한계는 남는다.

### 후속 실험 후보
- Ticket 단위 group split
- Ticket 피처를 적용한 RF/HGB/XGB 동일 조건 비교
- 여러 random seed에서 결과 안정성 확인
- 각 파생 피처의 ablation table 정리

이 항목들은 현재 결과를 무효화한다는 의미가 아니라, **어디까지 결과를 신뢰할 수 있고 다음에 무엇을 검증해야 하는지**를 기록한 것이다.
