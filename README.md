# Titanic Survival Prediction

Kaggle Titanic 데이터를 사용해 생존 여부를 예측하고, 데이터 누수를 피한 교차검증부터 내부 hold-out 평가와 Kaggle 제출 파일 생성까지 연결한 머신러닝 실험입니다.

## 핵심 결과

최종 후보는 `FamilySize`, `IsAlone`, `Title`, `TicketGroupSize`, `FarePerPerson`을 사용하는 튜닝 Random Forest입니다. 개발 데이터의 OOF 예측으로 threshold `0.47`을 선택했습니다.

| 평가 구간 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 5-Fold OOF | 0.834 | 0.766 | 0.817 | **0.791** | - |
| 내부 hold-out | 0.816 | 0.750 | 0.783 | **0.766** | **0.853** |

내부 hold-out은 모델·피처·threshold 선택이 끝난 뒤 한 번만 평가했습니다. Kaggle test 데이터에는 정답이 없으므로 성능 평가가 아니라 `submission.csv` 생성에만 사용합니다.

## 의사결정 흐름

| 구간 | 핵심 결정 | 근거 |
|---|---|---|
| 베이스라인 | LR·RF·HGB·XGB 비교 | 동일한 5-Fold에서 F1과 ROC-AUC를 함께 비교 |
| 피처 | `FamilySize`·`IsAlone`·`Title` 채택 | 가족 구성과 호칭이 승객의 생존 맥락을 보완 |
| 최종 후보 | Ticket 피처를 추가한 튜닝 RF | OOF F1 `0.791`로 실험 후보 중 가장 높음 |
| 최종 평가 | 내부 hold-out을 마지막에 1회 사용 | F1 `0.766`, ROC-AUC `0.853` |
| Kaggle 추론 | 전체 train 891명으로 재학습 | test 418명의 제출 예측 생성 |

![모델 비교](assets/model_comparison.png)

![내부 hold-out 혼동행렬](assets/confusion_matrix.png)

## 검증 설계

- `titanic_train.csv`를 개발 데이터 80%와 내부 hold-out 20%로 계층 분할했습니다.
- 개발 데이터에서 `StratifiedKFold` 5-Fold 교차검증을 수행했습니다.
- 결측치 처리, 스케일링, one-hot encoding과 Ticket 피처 계산은 Pipeline 안에서 Fold별로 학습했습니다.
- threshold는 개발 데이터 OOF 확률에서만 선택했습니다.
- 선택 완료 후 내부 hold-out을 한 번 평가하고, 전체 train으로 다시 학습해 Kaggle 제출 예측을 만들었습니다.

## Ticket 피처의 한계

교차검증에서 학습 Fold에 없던 티켓은 그룹 크기 1로 처리했습니다. 또한 승객 단위 무작위 분할을 사용했기 때문에 같은 티켓의 승객이 서로 다른 Fold에 배치될 수 있습니다. 따라서 티켓 동행 정보의 일반화 가능성에는 제약이 있으며, 후속 실험에서는 티켓 단위 분할을 검토할 수 있습니다.

## 저장소 구조

```text
titanic-survival-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── titanic_survival_modeling.ipynb
├── data/
│   └── README.md
└── assets/
    ├── model_comparison.png
    └── confusion_matrix.png
```

CSV와 `submission.csv`는 Git에서 제외됩니다.

## 실행 방법

Python 3.12 환경을 권장합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

[Kaggle Titanic 데이터 페이지](https://www.kaggle.com/competitions/titanic/data)에서 `train.csv`, `test.csv`를 내려받아 각각 다음 이름으로 배치합니다.

```text
data/titanic_train.csv
data/titanic_test.csv
```

설치된 Jupyter 또는 VS Code에서 `notebooks/titanic_survival_modeling.ipynb`를 열고 모든 셀을 순서대로 실행하면 다음 파일이 생성됩니다.

```text
assets/model_comparison.png
assets/confusion_matrix.png
submission.csv
```

