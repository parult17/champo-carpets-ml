# Champo Carpets ML Analysis

This repository contains the code and outputs for the case study:

**Champo Carpets: Improving Business-to-Business Sales using Machine Learning Algorithms**

The objective is to predict whether carpet samples sent to B2B customers convert into actual orders, and to use the model results to make business recommendations.

## Project Structure

```text
scripts/
  champo_utils.py
  part1_data_audit.py
  part2_eda.py
  part3_logistic_regression.py
  part4_tree_models.py
  part5_recommendations.py
  part6_report_inputs.py

outputs/
  Generated audit files, EDA tables, model metrics, and report inputs
```

## Main Report

The formatted Word report is available at:

```text
outputs/Champo_Carpets_Main_Report.docx
```

It follows the requested formatting: A4 page size, Times New Roman, 12-point font, and 1.5 line spacing.

## How To Run

Run the scripts in this order:

```powershell
python scripts/part1_data_audit.py
python scripts/part2_eda.py
python scripts/part3_logistic_regression.py
python scripts/part4_tree_models.py
python scripts/part5_recommendations.py
python scripts/part6_report_inputs.py
python scripts/build_main_report.py
```

## Dataset Note

The original Excel dataset is not included in this repository because it is part of the restricted case material.

The scripts currently expect the dataset at:

```text
D:\Masters' Union\TBM\Terms\Term 5\Data Driven\Final Project\IMB881-XLS-ENG.xlsx
```

## Main Output

The main summary file is:

```text
outputs/part6_report_inputs.md
```

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest

## Key Result

Logistic Regression performed best overall based on F1 score and interpretability. Important conversion drivers included `AreaFt`, `QtyRequired`, and product type.
