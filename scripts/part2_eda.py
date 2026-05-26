from __future__ import annotations

import pandas as pd

from champo_utils import load_sample_data, read_sheet, ORDER_SHEET, write_csv


def conversion_table(df: pd.DataFrame, column: str, min_rows: int = 20) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)["Order_Conversion"]
        .agg(records="count", conversions="sum", conversion_rate="mean")
        .reset_index()
    )
    grouped = grouped[grouped["records"] >= min_rows].copy()
    grouped["conversion_rate"] = grouped["conversion_rate"].round(4)
    return grouped.sort_values(["conversion_rate", "records"], ascending=[False, False])


def main() -> None:
    x, y = load_sample_data()
    sample = x.copy()
    sample["Order_Conversion"] = y

    for column in ["CountryName", "CustomerCode", "ITEM_NAME", "ShapeName"]:
        write_csv(conversion_table(sample, column), f"part2_conversion_by_{column}.csv")

    numeric_summary = (
        sample.groupby("Order_Conversion")[["QtyRequired", "AreaFt"]]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    numeric_summary.columns = ["_".join(col) for col in numeric_summary.columns]
    write_csv(numeric_summary.reset_index(), "part2_numeric_summary.csv")

    orders = read_sheet(ORDER_SHEET)
    order_summary = (
        orders.groupby("CustomerCode")
        .agg(order_rows=("CustomerCode", "count"), total_amount=("Amount", "sum"), total_area=("TotalArea", "sum"))
        .reset_index()
        .sort_values("total_amount", ascending=False)
        .head(15)
    )
    write_csv(order_summary, "part2_top_order_customers.csv")

    print("Part 2 complete")
    print("Created EDA CSVs in outputs/")
    print("Top conversion by country:")
    print(conversion_table(sample, "CountryName").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
