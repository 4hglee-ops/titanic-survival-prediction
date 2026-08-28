from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    title = result["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    result["Title"] = title.where(title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")
    return result


class TicketFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build ticket-derived features using counts learned only from the fit fold."""

    def fit(self, X, y=None):
        self.ticket_counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        result = X.copy()
        result["TicketGroupSize"] = result["Ticket"].map(self.ticket_counts_).fillna(1)
        result["FarePerPerson"] = result["Fare"] / result["TicketGroupSize"]
        return result.drop(columns="Ticket")


def make_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    fare_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, [
            "Age", "FamilySize", "IsAlone", "TicketGroupSize", "FarePerPerson",
        ]),
        ("fare", fare_transformer, ["Fare"]),
        ("cat", categorical_transformer, ["Pclass", "Sex", "Embarked", "Title"]),
    ])


def make_pipeline(model):
    return Pipeline([
        ("ticket_features", TicketFeatureBuilder()),
        ("preprocessor", make_preprocessor()),
        ("model", model),
    ])


def threshold_search(y_true, probabilities):
    rows = []
    for threshold in np.arange(0.20, 0.81, 0.01):
        pred = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    table = pd.DataFrame(rows)
    return table.loc[table["f1"].idxmax()]


def main():
    train_path = DATA_DIR / "titanic_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "data/titanic_train.csv가 필요합니다. Kaggle train.csv를 해당 이름으로 배치한 뒤 다시 실행하세요."
        )

    train_df = pd.read_csv(train_path)
    X = train_df.drop(columns="Survived")
    y = train_df["Survived"]

    # README/notebook과 동일한 locked hold-out 분리. 이 비교에서는 hold-out을 열지 않는다.
    X_dev, _X_holdout, y_dev, _y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_dev = add_context_features(X_dev)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # RF는 현재 notebook에서 선택된 튜닝 파라미터를 그대로 사용한다.
    models = {
        "Tuned RF": RandomForestClassifier(
            n_estimators=800,
            min_samples_split=15,
            min_samples_leaf=2,
            max_samples=None,
            max_features="sqrt",
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "HGB": HistGradientBoostingClassifier(random_state=SEED),
        "XGB": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    result_rows = []
    for name, model in models.items():
        pipeline = make_pipeline(model)
        proba = cross_val_predict(
            pipeline,
            X_dev,
            y_dev,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        best = threshold_search(y_dev, proba)
        result_rows.append({
            "model": name,
            **best.to_dict(),
            "roc_auc": roc_auc_score(y_dev, proba),
        })

    results = (
        pd.DataFrame(result_rows)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    results.to_csv(ASSET_DIR / "ticket_model_recomparison.csv", index=False)

    print("\nTicket feature fair comparison — development OOF only")
    print(results.round(4).to_string(index=False))
    print("\nSaved:", ASSET_DIR / "ticket_model_recomparison.csv")
    print("Locked hold-out was not used for model selection.")


if __name__ == "__main__":
    main()
