from __future__ import annotations

import numpy as np
import pandas as pd

from champo_utils import (
    build_tree,
    classification_metrics,
    prepare_model_matrix,
    predict_tree,
    standardize,
    train_test_split,
    tree_feature_counts,
    write_csv,
    write_json,
)


def predict_many(tree: dict, x: np.ndarray) -> np.ndarray:
    return np.array([predict_tree(tree, row) for row in x])


def main() -> None:
    x, y = prepare_model_matrix()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train_np, x_test_np = standardize(x_train, x_test)
    feature_names = list(x.columns)

    tree = build_tree(x_train_np, y_train.to_numpy(), feature_names, max_depth=4, min_samples=45)
    tree_probs = predict_many(tree, x_test_np)
    tree_metrics = classification_metrics(y_test, tree_probs)

    rng = np.random.default_rng(42)
    forest_probs = []
    forest_counts: dict[str, int] = {}
    max_features = max(4, int(np.sqrt(x_train_np.shape[1])))
    for _ in range(25):
        sample_idx = rng.choice(len(x_train_np), size=len(x_train_np), replace=True)
        forest_tree = build_tree(
            x_train_np[sample_idx],
            y_train.to_numpy()[sample_idx],
            feature_names,
            max_depth=5,
            min_samples=40,
            max_features=max_features,
            rng=rng,
        )
        forest_probs.append(predict_many(forest_tree, x_test_np))
        for feature, count in tree_feature_counts(forest_tree).items():
            forest_counts[feature] = forest_counts.get(feature, 0) + count

    rf_probs = np.mean(forest_probs, axis=0)
    rf_metrics = classification_metrics(y_test, rf_probs)

    model_comparison = pd.DataFrame(
        [
            {"model": "Decision Tree", **tree_metrics},
            {"model": "Random Forest", **rf_metrics},
        ]
    )
    importance = (
        pd.DataFrame({"feature": list(forest_counts), "split_count": list(forest_counts.values())})
        .sort_values("split_count", ascending=False)
        .head(25)
    )

    write_json({"decision_tree": tree}, "part4_decision_tree_structure.json")
    write_csv(model_comparison, "part4_tree_model_comparison.csv")
    write_csv(importance, "part4_random_forest_feature_importance.csv")

    print("Part 4 complete")
    print(model_comparison.to_string(index=False))
    print("Top RF features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
