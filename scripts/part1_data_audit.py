from __future__ import annotations

import pandas as pd

from champo_utils import DATA_PATH, OUTPUT_DIR, SAMPLE_SHEET, read_sheet, write_csv, write_json


def main() -> None:
    excel = pd.ExcelFile(DATA_PATH)
    rows = []
    for sheet in excel.sheet_names:
        df = read_sheet(sheet)
        rows.append(
            {
                "sheet": sheet,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    audit = pd.DataFrame(rows)
    write_csv(audit, "part1_workbook_audit.csv")

    sample = read_sheet(SAMPLE_SHEET)
    target = sample["Order_Conversion"].dropna().astype(int)
    target_summary = {
        "rows": int(sample.shape[0]),
        "target_counts": target.value_counts().sort_index().to_dict(),
        "conversion_rate": round(float(target.mean()), 4),
        "columns": list(sample.columns),
    }
    write_json(target_summary, "part1_sample_target_summary.json")

    print("Part 1 complete")
    print(audit.to_string(index=False))
    print("Sample conversion rate:", target_summary["conversion_rate"])
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
