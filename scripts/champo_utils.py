from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = Path(
    r"D:\Masters' Union\TBM\Terms\Term 5\Data Driven\Final Project\IMB881-XLS-ENG.xlsx"
)

SAMPLE_SHEET = "Data on Sample ONLY"
ORDER_SHEET = "Data Order ONLY"
RECOMMENDATION_SHEET = "Data for Recommendation"
CLUSTERING_SHEET = "Data for Clustering"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_")
        if c is not None and not str(c).startswith("Unnamed")
        else f"unnamed_{i}"
        for i, c in enumerate(df.columns)
    ]
    return df


def read_sheet(sheet_name: str) -> pd.DataFrame:
    return clean_columns(pd.read_excel(DATA_PATH, sheet_name=sheet_name))


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def write_json(data: dict, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    df = read_sheet(SAMPLE_SHEET)
    target_col = "Order_Conversion"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int)
    x = df.drop(columns=[target_col])
    return x, y


def prepare_model_matrix() -> tuple[pd.DataFrame, pd.Series]:
    x, y = load_sample_data()

    # Avoid duplicate signal: use CountryName, not both CountryName and one-hot country flags.
    country_flags = ["USA", "UK", "Italy", "Belgium", "Romania", "Australia", "India"]
    drop_cols = [c for c in country_flags if c in x.columns]
    x = x.drop(columns=drop_cols)

    for col in x.columns:
        if x[col].dtype == "object":
            x[col] = x[col].fillna("Unknown").astype(str).str.strip()
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    x_encoded = pd.get_dummies(x, drop_first=False)
    return x_encoded.astype(float), y


def train_test_split(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for value in sorted(y.unique()):
        indices = np.array(y[y == value].index)
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * test_size))
        test_idx.extend(indices[:cut])
        train_idx.extend(indices[cut:])
    return x.loc[train_idx], x.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, 1)
    return ((train - mean) / std).to_numpy(), ((test - mean) / std).to_numpy()


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 1200,
    l2: float = 0.01,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x_train.shape[1])
    bias = 0.0
    n = len(y_train)

    for _ in range(epochs):
        probs = sigmoid(x_train @ weights + bias)
        error = probs - y_train
        grad_w = (x_train.T @ error) / n + l2 * weights
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def classification_metrics(y_true: pd.Series | np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0
    p = y.mean()
    return 1 - p**2 - (1 - p) ** 2


def build_tree(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_depth: int = 4,
    min_samples: int = 40,
    max_features: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng(42)
    prob = float(y.mean())
    node = {"prob": prob, "n": int(len(y))}

    if max_depth == 0 or len(y) < min_samples or gini(y) == 0:
        return node

    feature_indices = np.arange(x.shape[1])
    if max_features:
        feature_indices = rng.choice(feature_indices, size=min(max_features, x.shape[1]), replace=False)

    best = None
    parent_gini = gini(y)
    for j in feature_indices:
        values = np.unique(x[:, j])
        if len(values) <= 1:
            continue
        thresholds = [0.5] if set(values).issubset({0, 1}) else np.quantile(values, [0.25, 0.5, 0.75])
        for threshold in thresholds:
            left = x[:, j] <= threshold
            right = ~left
            if left.sum() < min_samples or right.sum() < min_samples:
                continue
            weighted = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / len(y)
            gain = parent_gini - weighted
            if best is None or gain > best["gain"]:
                best = {"feature": j, "threshold": float(threshold), "gain": float(gain), "left": left, "right": right}

    if best is None:
        return node

    node.update({"feature": int(best["feature"]), "feature_name": feature_names[best["feature"]], "threshold": best["threshold"]})
    node["left"] = build_tree(x[best["left"]], y[best["left"]], feature_names, max_depth - 1, min_samples, max_features, rng)
    node["right"] = build_tree(x[best["right"]], y[best["right"]], feature_names, max_depth - 1, min_samples, max_features, rng)
    return node


def predict_tree(tree: dict, row: np.ndarray) -> float:
    node = tree
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return node["prob"]


def tree_feature_counts(tree: dict, counts: dict[str, int] | None = None) -> dict[str, int]:
    counts = counts or {}
    if "feature_name" in tree:
        counts[tree["feature_name"]] = counts.get(tree["feature_name"], 0) + 1
        tree_feature_counts(tree["left"], counts)
        tree_feature_counts(tree["right"], counts)
    return counts
