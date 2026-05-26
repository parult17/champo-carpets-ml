# Champo Carpets Report Inputs

## Problem
Champo sends costly carpet samples to B2B customers. The business question is which samples are likely to convert into orders.

## Data
- Sample-only rows used for ML: 5820
- Overall conversion rate: 20.1%

## Model Performance
- Logistic Regression: accuracy 0.8426, precision 0.7358, recall 0.3348, F1 0.4602
- Decision Tree: accuracy 0.8366, precision 0.7416, recall 0.2833, F1 0.4099
- Random Forest: accuracy 0.834, precision 0.8333, recall 0.2146, F1 0.3413

## Logistic Regression Drivers
| feature | coefficient |
| --- | --- |
| AreaFt | 0.9145 |
| Other | 0.4911 |
| Knotted | 0.3729 |
| QtyRequired | 0.184 |
| Hand_Tufted | -0.1768 |
| Double_Back | -0.1309 |
| Square | 0.0673 |
| Hand_Woven | -0.0468 |

## Random Forest Drivers
| feature | split_count |
| --- | --- |
| AreaFt | 53 |
| QtyRequired | 28 |
| Knotted | 22 |
| Other | 20 |
| Hand_Tufted | 18 |
| Durry | 15 |
| Hand_Woven | 12 |
| Double_Back | 11 |

## Recommendation Bullets
- Score future samples before production and prioritize high-probability opportunities.
- Reduce sample effort for low-probability customer/product combinations unless strategic value is high.
- Segment customers using order value, area, quantity, and product mix before sales targeting.
- Use explainable model drivers in the report so recommendations are business-friendly.