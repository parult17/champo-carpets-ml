from __future__ import annotations

import numpy as np
import pandas as pd

from champo_utils import (
    classification_metrics,
    fit_logistic_regression,
    prepare_model_matrix,
    sigmoid,
    standardize,
    train_test_split,
    write_csv,
    write_json,
)


def main() -> None:
    x, y = prepare_model_matrix()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train_std, x_test_std = standardize(x_train, x_test)

    weights, bias = fit_logistic_regression(x_train_std, y_train.to_numpy())
    test_probs = sigmoid(x_test_std @ weights + bias)
    metrics = classification_metrics(y_test, test_probs)

    coefficients = pd.DataFrame({"feature": x.columns, "coefficient": weights})
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)

    write_json({"model": "Logistic Regression", "metrics": metrics}, "part3_logistic_metrics.json")
    write_csv(coefficients.head(25), "part3_logistic_top_coefficients.csv")

    scored = pd.DataFrame({"actual": y_test.to_numpy(), "predicted_probability": np.round(test_probs, 4)})
    scored["predicted"] = (scored["predicted_probability"] >= 0.5).astype(int)
    write_csv(scored, "part3_logistic_test_predictions.csv")

    print("Part 3 complete")
    print(metrics)
    print(coefficients.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
