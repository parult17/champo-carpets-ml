from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from champo_utils import OUTPUT_DIR


def read_json(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def main() -> None:
    target = read_json("part1_sample_target_summary.json")
    lr = read_json("part3_logistic_metrics.json")["metrics"]
    tree_models = pd.read_csv(OUTPUT_DIR / "part4_tree_model_comparison.csv")
    rf_features = pd.read_csv(OUTPUT_DIR / "part4_random_forest_feature_importance.csv").head(8)
    lr_features = pd.read_csv(OUTPUT_DIR / "part3_logistic_top_coefficients.csv").head(8)

    lines = [
        "# Champo Carpets Report Inputs",
        "",
        "## Problem",
        "Champo sends costly carpet samples to B2B customers. The business question is which samples are likely to convert into orders.",
        "",
        "## Data",
        f"- Sample-only rows used for ML: {target['rows']}",
        f"- Overall conversion rate: {target['conversion_rate']:.1%}",
        "",
        "## Model Performance",
        f"- Logistic Regression: accuracy {lr['accuracy']}, precision {lr['precision']}, recall {lr['recall']}, F1 {lr['f1']}",
    ]

    for _, row in tree_models.iterrows():
        lines.append(
            f"- {row['model']}: accuracy {row['accuracy']}, precision {row['precision']}, recall {row['recall']}, F1 {row['f1']}"
        )

    lines += [
        "",
        "## Logistic Regression Drivers",
        markdown_table(lr_features[["feature", "coefficient"]].round(4)),
        "",
        "## Random Forest Drivers",
        markdown_table(rf_features[["feature", "split_count"]]),
        "",
        "## Recommendation Bullets",
        "- Score future samples before production and prioritize high-probability opportunities.",
        "- Reduce sample effort for low-probability customer/product combinations unless strategic value is high.",
        "- Segment customers using order value, area, quantity, and product mix before sales targeting.",
        "- Use explainable model drivers in the report so recommendations are business-friendly.",
    ]

    path = OUTPUT_DIR / "part6_report_inputs.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Part 6 complete")
    print(path)


if __name__ == "__main__":
    main()
