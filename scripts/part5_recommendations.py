from __future__ import annotations

import pandas as pd

from champo_utils import (
    CLUSTERING_SHEET,
    RECOMMENDATION_SHEET,
    load_sample_data,
    read_sheet,
    write_csv,
    write_json,
)


def main() -> None:
    x, y = load_sample_data()
    sample = x.copy()
    sample["Order_Conversion"] = y

    customer_conversion = (
        sample.groupby("CustomerCode")["Order_Conversion"]
        .agg(sample_count="count", conversions="sum", conversion_rate="mean")
        .reset_index()
    )
    customer_conversion = customer_conversion[customer_conversion["sample_count"] >= 20]
    customer_conversion["conversion_rate"] = customer_conversion["conversion_rate"].round(4)

    recommendation = read_sheet(RECOMMENDATION_SHEET).rename(columns={"Customer": "CustomerCode", "Customer_": "CustomerCode"})
    if "CustomerCode" not in recommendation.columns:
        recommendation = recommendation.rename(columns={recommendation.columns[0]: "CustomerCode"})

    clustering = read_sheet(CLUSTERING_SHEET).rename(columns={"Row_Labels": "CustomerCode"})
    merged = customer_conversion.merge(recommendation, on="CustomerCode", how="left")
    merged = merged.merge(clustering, on="CustomerCode", how="left", suffixes=("_rec", "_cluster"))

    focus_customers = merged.sort_values(["conversion_rate", "sample_count"], ascending=[False, False]).head(15)
    write_csv(focus_customers, "part5_customer_recommendation_inputs.csv")

    playbook = {
        "recommendations": [
            "Prioritize sample follow-up for customer/product/shape combinations with historically higher conversion.",
            "Use low-conversion segments to reduce costly sample creation or require better pre-qualification.",
            "Use Random Forest feature importance for operational targeting, and Logistic Regression coefficients for explainable business storytelling.",
            "Cluster customers by quantity, area, amount, and product mix before assigning sales strategies.",
            "For A-11, use association-rule colors as bundling/design cues rather than a full-company recommendation."
        ]
    }
    write_json(playbook, "part5_recommendation_playbook.json")

    print("Part 5 complete")
    print(focus_customers[["CustomerCode", "sample_count", "conversion_rate"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
