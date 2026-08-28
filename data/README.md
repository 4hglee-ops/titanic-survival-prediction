# Data

이 프로젝트는 Kaggle의 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) 데이터를 사용합니다.

Kaggle에서 데이터를 내려받아 이 디렉터리에 다음 이름으로 배치합니다.

```text
data/
├── titanic_train.csv  # Kaggle train.csv
└── titanic_test.csv   # Kaggle test.csv
```

`titanic_train.csv`에는 정답 열 `Survived`가 있지만 `titanic_test.csv`에는 없습니다. 후자는 성능 검증이 아니라 Kaggle 제출 예측 생성에 사용합니다.

CSV 파일은 저장소에 포함하지 않습니다. 데이터 사용 조건은 Kaggle 대회 규칙을 따릅니다.

