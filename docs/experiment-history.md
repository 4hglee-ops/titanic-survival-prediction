# Experiment History

이 문서는 Titanic 생존 예측 프로젝트에서 **실제로 거친 실험 흐름**을 시간 순서에 가깝게 정리합니다.

정확한 수치가 과거 기록에서 확인되는 경우에만 적고, 확인되지 않는 실험은 시도 사실만 기록합니다. 마지막의 후속 아이디어는 실제 수행 결과와 구분합니다.

---

## 1. 기본 모델 비교

먼저 원본 중심 피처를 사용해 여러 분류 모델을 동일한 교차검증 기준에서 비교했다.

현재 재구성 노트북의 기본 피처 결과는 다음과 같다.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.791 | 0.748 | 0.692 | 0.717 | 0.858 |
| Random Forest | 0.802 | 0.756 | 0.718 | 0.736 | 0.863 |
| HistGradientBoosting | 0.820 | 0.796 | 0.718 | 0.755 | 0.877 |
| XGBoost | **0.824** | **0.805** | **0.721** | **0.760** | **0.883** |

### 관찰
기본 피처에서는 XGBoost가 F1과 ROC-AUC 모두 가장 높았다. 따라서 이 단계에서 RF를 최종 모델로 확정하지 않았다.

---

## 2. 승객 맥락 피처 추가

개별 승객의 원본 컬럼만으로는 가족 구성과 사회적 역할을 충분히 표현하지 못한다고 보고 다음 피처를 만들었다.

- `FamilySize`
- `IsAlone`
- `Title`

현재 재구성 노트북에서 이 피처들을 추가한 결과는 다음과 같다.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.820 | 0.786 | 0.736 | 0.759 | 0.871 |
| Random Forest | 0.802 | 0.742 | 0.744 | 0.742 | 0.870 |
| HistGradientBoosting | **0.833** | **0.807** | **0.744** | **0.773** | 0.886 |
| XGBoost | 0.829 | 0.806 | 0.733 | 0.766 | **0.894** |

### 관찰
- LR, HGB, XGB의 F1이 기본 피처 대비 상승했다.
- HGB는 F1 기준 가장 높았다.
- XGB는 ROC-AUC 기준 가장 높았다.

이 결과만 보면 HGB/XGB가 유력했지만 실험은 여기서 끝내지 않았다.

---

## 3. 추가 Feature Engineering 탐색

과거 학습 채팅에서 다음 피처들을 추가로 검토한 기록이 남아 있다.

### Name / Title
이름 전체를 그대로 범주형 변수로 사용하지 않고 `Mr`, `Mrs`, `Miss`, `Master` 등의 호칭을 추출했다. 희귀 호칭은 `Rare`로 묶었다.

### Age 보정
단순 전체 중앙값 대신 `Title`별 연령 분포를 활용해 결측치를 처리하는 실험을 진행했다.

### CabinKnown
객실 번호 자체보다 Cabin 정보가 존재하는지 여부가 유용할 수 있다는 가설을 시험했다.

과거 기록에서:

- HGB + `FamilySize` + `IsAlone` + `Title`: F1 약 **0.7733**
- 위 구성 + `CabinKnown`: F1 약 **0.7662**

성능이 하락해 `CabinKnown`은 최종 피처에서 제외했다.

### Surname / FamilySurvival
성(last name)을 추출하고 가족 단위 생존 정보를 활용할 수 있는지 실험했다.

### TicketGroupSize
같은 Ticket을 공유하는 승객 수를 이용해 동행 그룹 규모를 표현했다.

### FarePerPerson
Ticket 그룹 크기를 이용해 요금을 승객 단위로 보정하는 피처를 실험했다.

### AgeGroup / Mother
연령대와 어머니 여부처럼 Titanic 도메인에서 의미가 있을 수 있는 조건형 피처도 후보로 실험했다.

> 이 피처들의 개별 ablation 수치는 현재 복원된 과거 기록에서 모두 확인되지는 않는다. 따라서 최종 채택 여부를 과장하지 않고 실험 후보로만 기록한다.

---

## 4. Random Forest 튜닝

Random Forest에 대해 `RandomizedSearchCV`를 적용해 다음과 같은 과적합 제어 파라미터를 탐색했다.

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`
- `max_samples`

현재 재구성 노트북에서 선택된 파라미터 예시는 다음과 같다.

```text
n_estimators = 800
max_depth = 12
min_samples_split = 15
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced_subsample
max_samples = None
```

현재 재구성 과정에서 튜닝 RF의 best CV F1은 약 `0.774`였다.

---

## 5. 모델별 threshold 재탐색

과거 학습 과정에서 단순 `0.5` threshold를 고정하지 않고 RF, HGB, XGB 각각의 OOF probability에서 F1이 가장 높은 threshold를 탐색했다.

당시 기록에서 복원된 비교는 다음과 같다.

| Model | Threshold | F1 |
|---|---:|---:|
| Tuned Random Forest | **0.46** | **0.7852** |
| HistGradientBoosting | 0.53 | 0.7814 |
| XGBoost | 0.42 | 0.7761 |

RF threshold `0.46`의 세부 결과:

- Precision: 약 `0.7559`
- Recall: 약 `0.8169`
- F1: 약 `0.7852`

### 당시 결론
**RF + Title + FamilySize + IsAlone, threshold 0.46**을 F1 기준 최종 후보로 선택했다.

이 기록은 RF가 기본 모델 비교에서 처음부터 최고였기 때문에 선택된 것이 아니라, **튜닝과 threshold 재비교 이후 최종적으로 F1이 가장 높았기 때문에 선택되었다는 점**을 보여준다.

---

# GitHub 재검증 및 정리 단계

2026-08-27~28에 기존 학습 프로젝트를 GitHub에 정리하면서 단순히 과거 노트북을 업로드하지 않고 검증 구조를 다시 점검했다.

---

## 6. 내부 hold-out을 먼저 분리

`train.csv` 891명을 다음과 같이 분리했다.

```text
전체 train 891
├─ development 712 (80%)
└─ hold-out 179 (20%)
```

- 계층 분할 사용
- EDA와 모델 선택은 development에 한정
- hold-out은 마지막 평가 전까지 사용하지 않음

---

## 7. 전처리 누수 방지

다음 전처리를 `Pipeline` / `ColumnTransformer` 내부로 이동했다.

- 결측치 처리
- 표준화
- `Fare` log 변환
- one-hot encoding

따라서 각 CV Fold에서 학습 Fold의 정보로만 전처리기가 fit된다.

---

## 8. Ticket 피처를 Fold-safe하게 재구현

과거 Ticket 기반 피처 아이디어를 그대로 전체 데이터에서 계산하지 않고 `TicketFeatureBuilder`라는 transformer로 구현했다.

동작 방식:

1. 학습 Fold에서만 `Ticket` 빈도를 계산한다.
2. validation Fold에 매핑한다.
3. 처음 보는 Ticket은 그룹 크기 `1`로 처리한다.
4. `FarePerPerson = Fare / TicketGroupSize`를 계산한다.

이를 통해 Ticket 통계가 validation Fold 전체 정보를 미리 보는 문제를 줄였다.

---

## 9. OOF threshold `0.47`

재구성된 튜닝 RF + 맥락 피처 + Ticket 피처에서 개발 데이터 OOF probability를 사용해 threshold를 다시 탐색했다.

최종 선택:

```text
threshold = 0.47
Accuracy  = 0.834
Precision = 0.766
Recall    = 0.817
F1        = 0.791
```

과거 학습 당시 RF 결과 `0.7852`보다 F1이 소폭 상승했다.

단, 과거 실험과 현재 재검증은 전처리와 피처 구현이 완전히 동일한 실험이 아니므로 단순한 점수 상승만으로 직접 우열을 해석하지 않는다.

---

## 10. Locked hold-out 최종 평가

모델, 피처, threshold 선택을 모두 마친 뒤 179명의 hold-out을 한 번 평가했다.

| Metric | Score |
|---|---:|
| Accuracy | 0.816 |
| Precision | 0.750 |
| Recall | 0.783 |
| F1 | **0.766** |
| ROC-AUC | **0.853** |

OOF 결과보다 낮았지만 이 결과에 맞춰 모델을 다시 선택하거나 threshold를 수정하지 않았다.

---

## 11. 전체 train 재학습과 Kaggle inference

최종 검증 후 891명 전체 train으로 모델을 다시 학습하고 Kaggle test 418명을 예측한다.

Kaggle test에는 정답이 없기 때문에 이 단계는 성능 평가가 아니라 제출 예측 생성 단계다.

---

# 실험에서 배운 점

1. **피처가 그럴듯하다는 것과 실제 도움이 된다는 것은 다르다.** `CabinKnown`처럼 가설은 타당해도 CV 성능이 떨어질 수 있다.
2. **기본 모델 순위가 최종 모델 순위와 같지 않을 수 있다.** 튜닝과 threshold 조정을 포함하면 RF가 최종 F1에서 앞섰다.
3. **threshold도 모델 선택의 일부다.** 0.5를 관습적으로 고정하지 않고 OOF 기반으로 판단했다.
4. **좋은 피처도 검증 구현이 중요하다.** Ticket 그룹 피처는 계산 방법에 따라 데이터 누수 위험이 있다.
5. **hold-out 결과가 낮다고 다시 맞추지 않는 것도 의사결정이다.** 마지막 검증 세트의 역할을 유지해야 실제 일반화 성능을 더 정직하게 볼 수 있다.

# 추가로 검토할 수 있었던 항목 — 미수행

아래 항목은 프로젝트를 정리하면서 확인한 후속 아이디어이며, **이 저장소에서 실제로 수행한 실험은 아니다.**

- Ticket 기반 그룹 분할과 현재 승객 단위 분할 비교
- 동일한 Ticket 피처 조건에서 RF / HGB / XGB 재비교
- 피처별 ablation table 작성
- 여러 random seed에서 평균과 표준편차 비교
- probability calibration 및 threshold 안정성 확인

이 프로젝트의 마무리 기준은 위 실험을 새로 수행하는 것이 아니라, **당시 학습 과정과 2026-08-27~28 재검증에서 실제로 수행한 범위를 정확하게 보존하는 것**이다.
