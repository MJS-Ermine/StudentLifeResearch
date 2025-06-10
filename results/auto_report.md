# StudentLife Auto Analysis Report

## 1. Research Background
This report is based on the StudentLife dataset, exploring the relationship between college students' daily behaviors, psychological states, and academic performance, and using multiple machine learning models for GPA prediction and feature importance analysis.

## 2. Model Performance Comparison
| Model | R2 | RMSE | MAE |
|-------|----|------|-----|
| LinearRegression | -2.052 | 0.651 | 0.531 |
| Ridge | -1.360 | 0.572 | 0.490 |
| RandomForest | 0.208 | 0.331 | 0.275 |

![](figures/correlation_heatmap.png)

### LinearRegression Feature Importance
![](figures/feature_importance_LinearRegression.png)

### Ridge Feature Importance
![](figures/feature_importance_Ridge.png)

### RandomForest Feature Importance
![](figures/feature_importance_RandomForest.png)

## 3. Initial Findings Summary
- Some behavioral features (such as sleep, phone usage, activity, and psychological state) have significant predictive power for GPA.
- Different models have slightly different rankings of feature importance, suggesting further theoretical interpretation and model optimization.
- Detailed distribution graphs and feature correlations can be found in the results/figures/ directory.
