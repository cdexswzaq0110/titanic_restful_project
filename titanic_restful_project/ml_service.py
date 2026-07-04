import io
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORY_COLUMNS = ["Sex", "Embarked"]
MODEL_CHOICES = {
    "logistic_regression": LogisticRegression(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "extra_trees": ExtraTreesClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}
SEARCH_SPACES = {
    "logistic_regression": {
        "quick": {"model__C": [1.0], "model__solver": ["liblinear"], "model__max_iter": [500]},
        "balanced": {"model__C": [0.1, 1, 10], "model__solver": ["liblinear"], "model__max_iter": [500]},
        "full": {"model__C": [0.01, 0.1, 1, 10], "model__solver": ["liblinear", "lbfgs"], "model__max_iter": [500, 1000]},
    },
    "random_forest": {
        "quick": {"model__n_estimators": [100], "model__max_depth": [None, 8], "model__min_samples_split": [2], "model__min_samples_leaf": [1]},
        "balanced": {"model__n_estimators": [100, 200], "model__max_depth": [None, 6, 10], "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]},
        "full": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4]},
    },
    "gradient_boosting": {
        "quick": {"model__n_estimators": [100], "model__learning_rate": [0.1], "model__max_depth": [3]},
        "balanced": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
        "full": {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4]},
    },
    "extra_trees": {
        "quick": {"model__n_estimators": [200], "model__max_depth": [8], "model__min_samples_leaf": [2]},
        "balanced": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 12], "model__min_samples_leaf": [1, 2, 4]},
        "full": {"model__n_estimators": [200, 400, 600], "model__max_depth": [None, 8, 12, 16], "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.8]},
    },
    "svm": {
        "quick": {"model__C": [1], "model__kernel": ["rbf"], "model__gamma": ["scale"]},
        "balanced": {"model__C": [0.5, 1, 2], "model__kernel": ["rbf"], "model__gamma": ["scale", "auto"]},
        "full": {"model__C": [0.1, 0.5, 1, 2, 5], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
    },
}

_state_lock = threading.Lock()
_training_state = {"status": "idle", "result": None, "error": None}


def _read_data(database_path):
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(
            f"SELECT {', '.join(FEATURE_COLUMNS)}, Survived FROM titanic", connection
        )


def dashboard(database_path):
    data = _read_data(database_path)
    survived = int(data["Survived"].sum())
    total = len(data)
    return {
        "total": total,
        "survived": survived,
        "not_survived": total - survived,
        "survival_rate": round(survived / total, 4) if total else 0,
        "missing_values": {key: int(value) for key, value in data.isna().sum().items()},
        "overview": {
            "pclass": data["Pclass"].value_counts().sort_index().to_dict(),
            "sex": data["Sex"].value_counts().to_dict(),
            "embarked": data["Embarked"].fillna("Unknown").value_counts().to_dict(),
            "average_age": _safe_round(data["Age"].mean()),
            "average_fare": _safe_round(data["Fare"].mean()),
        },
    }


def _safe_round(value):
    return None if pd.isna(value) else round(float(value), 2)


def _pipeline(model_name):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("category", category, CATEGORY_COLUMNS),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", MODEL_CHOICES[model_name])])


def start_training(database_path, models_dir, model_name, search_mode):
    if model_name not in MODEL_CHOICES:
        raise ValueError("不支援的模型")
    if search_mode not in {"quick", "balanced", "full"}:
        raise ValueError("不支援的搜尋模式")

    with _state_lock:
        if _training_state["status"] == "training":
            raise RuntimeError("模型正在訓練中")
        _training_state.update(status="training", result=None, error=None)

    thread = threading.Thread(
        target=_train,
        args=(Path(database_path), Path(models_dir), model_name, search_mode),
        daemon=True,
    )
    thread.start()


def _train(database_path, models_dir, model_name, search_mode):
    started = time.perf_counter()
    try:
        data = _read_data(database_path)
        train_x, test_x, train_y, test_y = train_test_split(
            data[FEATURE_COLUMNS], data["Survived"], test_size=0.2, random_state=42, stratify=data["Survived"]
        )
        search = GridSearchCV(
            _pipeline(model_name),
            SEARCH_SPACES[model_name][search_mode],
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        search.fit(train_x, train_y)
        predictions = search.predict(test_x)
        probabilities = search.predict_proba(test_x)[:, 1]
        metrics = {
            "accuracy": round(accuracy_score(test_y, predictions), 4),
            "precision": round(precision_score(test_y, predictions, zero_division=0), 4),
            "recall": round(recall_score(test_y, predictions, zero_division=0), 4),
            "f1": round(f1_score(test_y, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(test_y, probabilities), 4),
            "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        }
        models_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_info = "Median imputation, most-frequent categorical imputation, one-hot encoding, numeric scaling"
        equivalent = _find_equivalent_model(models_dir, model_name, search_mode, search.best_params_, metrics, preprocessing_info)
        if equivalent:
            if not (models_dir / "active_model.json").exists():
                _write_active(models_dir, equivalent)
            with _state_lock:
                _training_state.update(status="completed", result={**equivalent, "reused": True}, error=None)
            return
        version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        joblib_name = f"titanic_model_{version}.joblib"
        metadata_name = f"titanic_model_{version}.json"
        joblib.dump(search.best_estimator_, models_dir / joblib_name)
        metadata = {
            "version": version,
            "model_name": model_name,
            "model_path": str(Path("models") / joblib_name),
            "metadata_path": str(Path("models") / metadata_name),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "search_mode": search_mode,
            "best_params": search.best_params_,
            "metrics": metrics,
            "training_seconds": round(time.perf_counter() - started, 2),
            "feature_columns": FEATURE_COLUMNS,
            "preprocessing_info": preprocessing_info,
        }
        (models_dir / metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        active_path = models_dir / "active_model.json"
        if not active_path.exists():
            _write_active(models_dir, metadata)
        with _state_lock:
            _training_state.update(status="completed", result=metadata, error=None)
    except Exception as error:
        with _state_lock:
            _training_state.update(status="failed", result=None, error=str(error))


def training_status():
    with _state_lock:
        return dict(_training_state)


def _find_equivalent_model(models_dir, model_name, search_mode, best_params, metrics, preprocessing_info):
    for path in Path(models_dir).glob("titanic_model_*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("model_name") == model_name
            and item.get("search_mode") == search_mode
            and item.get("best_params") == best_params
            and item.get("metrics") == metrics
            and item.get("feature_columns") == FEATURE_COLUMNS
            and item.get("preprocessing_info") == preprocessing_info
        ):
            return item
    return None


def list_models(models_dir):
    models_dir = Path(models_dir)
    active = _active_metadata(models_dir, required=False)
    active_version = active.get("version") if active else None
    items = []
    for path in sorted(models_dir.glob("titanic_model_*.json"), reverse=True):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["active"] = item["version"] == active_version
        items.append(item)
    if items:
        champion = max(items, key=lambda item: item["metrics"]["roc_auc"])["version"]
        for item in items:
            item["champion"] = item["version"] == champion
    return items


def activate_model(models_dir, version):
    item = next((item for item in list_models(models_dir) if item["version"] == version), None)
    if item is None:
        raise FileNotFoundError("找不到指定模型版本")
    _write_active(Path(models_dir), item)
    return item


def _write_active(models_dir, metadata):
    active = {"version": metadata["version"], "model_path": metadata["model_path"]}
    (models_dir / "active_model.json").write_text(json.dumps(active, indent=2), encoding="utf-8")


def _active_metadata(models_dir, required=True):
    active_path = Path(models_dir) / "active_model.json"
    if not active_path.exists():
        if required:
            raise FileNotFoundError("尚無啟用模型，請先訓練模型")
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    metadata_path = Path(models_dir) / f"titanic_model_{active['version']}.json"
    if not metadata_path.exists():
        raise FileNotFoundError("啟用模型的 metadata 不存在")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_active(models_dir):
    metadata = _active_metadata(models_dir)
    model_path = Path(models_dir) / Path(metadata["model_path"]).name
    if not model_path.exists():
        raise FileNotFoundError("啟用模型檔案不存在")
    return joblib.load(model_path), metadata


def validate_passenger(data):
    missing = [column for column in FEATURE_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"缺少欄位: {', '.join(missing)}")
    try:
        passenger = {
            "Pclass": int(data["Pclass"]),
            "Sex": str(data["Sex"]).lower(),
            "Age": float(data["Age"]),
            "SibSp": int(data["SibSp"]),
            "Parch": int(data["Parch"]),
            "Fare": float(data["Fare"]),
            "Embarked": str(data["Embarked"]).upper(),
        }
    except (TypeError, ValueError) as error:
        raise ValueError("數值欄位格式錯誤") from error
    if passenger["Pclass"] not in {1, 2, 3} or passenger["Sex"] not in {"male", "female"} or passenger["Embarked"] not in {"C", "Q", "S"}:
        raise ValueError("Pclass、Sex 或 Embarked 值不合法")
    if passenger["Age"] < 0 or passenger["Age"] > 120 or passenger["SibSp"] < 0 or passenger["Parch"] < 0 or passenger["Fare"] < 0:
        raise ValueError("年齡、同行人數與票價不可超出合理範圍")
    return passenger


def predict(models_dir, data):
    passenger = validate_passenger(data)
    model, metadata = _load_active(models_dir)
    probability = float(model.predict_proba(pd.DataFrame([passenger]))[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction else "Not Survived",
        "survival_probability": round(probability, 4),
        "risk_level": "Low" if probability >= 0.7 else "Medium" if probability >= 0.4 else "High",
        "model_version": metadata["version"],
        "model_name": metadata["model_name"],
        "explanation": _explain(passenger, probability),
    }


def _explain(passenger, probability):
    reasons = []
    if passenger["Pclass"] == 3:
        reasons.append("3 等艙在歷史資料中的存活率較低")
    if passenger["Sex"] == "male":
        reasons.append("男性在 Titanic 歷史資料中的存活率較低")
    if passenger["Fare"] < 15:
        reasons.append("較低票價可能對應較不利的艙位條件")
    if not reasons:
        reasons.append("艙等、性別與票價條件未出現主要低存活率規則")
    return f"模型預估生存率為 {probability:.1%}。" + "；".join(reasons) + "。"


def what_if(models_dir, data):
    passenger = validate_passenger(data)
    original = predict(models_dir, passenger)["survival_probability"]
    scenarios = [
        ("Pclass 改為 1", {"Pclass": 1}),
        ("Fare 改為 80", {"Fare": 80}),
        ("Embarked 改為 C", {"Embarked": "C"}),
    ]
    results = []
    for label, change in scenarios:
        changed = {**passenger, **change}
        probability = predict(models_dir, changed)["survival_probability"]
        results.append({"condition": label, "original": original, "changed": probability, "difference": round(probability - original, 4)})
    return results


def predict_csv(models_dir, uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as error:
        raise ValueError("無法讀取 CSV") from error
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"CSV 缺少欄位: {', '.join(missing)}")
    model, metadata = _load_active(models_dir)
    try:
        probabilities = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    except Exception as error:
        raise ValueError(f"CSV 資料格式錯誤: {error}") from error
    output = data.copy()
    output["prediction"] = ["Survived" if value >= 0.5 else "Not Survived" for value in probabilities]
    output["survival_probability"] = probabilities.round(4)
    output["risk_level"] = ["Low" if value >= 0.7 else "Medium" if value >= 0.4 else "High" for value in probabilities]
    output["model_version"] = metadata["version"]
    stream = io.BytesIO(output.to_csv(index=False).encode("utf-8-sig"))
    stream.seek(0)
    return stream


def feature_importance(models_dir):
    pipeline, metadata = _load_active(models_dir)
    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    names = preprocessing.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        return {"model_version": metadata["version"], "items": []}
    items = sorted(
        ({"feature": name.split("__", 1)[-1], "importance": round(float(value), 4)} for name, value in zip(names, values)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {"model_version": metadata["version"], "items": items}
