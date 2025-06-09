# StudentLife 自動化初步分析報告

## 1. 研究背景
本報告基於 StudentLife dataset，探討大學生日常行為、心理狀態與學業表現之間的關聯，並以多種機器學習模型進行 GPA 預測與特徵重要性分析。

## 2. 模型效能比較
| Model | R2 | RMSE | MAE |
|-------|----|------|-----|
| LinearRegression | 1.000 | 0.000 | 0.000 |
| Ridge | 0.949 | 0.084 | 0.071 |
| RandomForest | 0.968 | 0.067 | 0.057 |

![](figures/correlation_heatmap.png)

### LinearRegression 特徵重要性
![](figures/feature_importance_LinearRegression.png)

### Ridge 特徵重要性
![](figures/feature_importance_Ridge.png)

### RandomForest 特徵重要性
![](figures/feature_importance_RandomForest.png)

## 3. 初步發現摘要
- 部分行為特徵（如睡眠、手機使用、活動、心理狀態）對 GPA 具顯著預測力。
- 不同模型對特徵重要性的排序略有差異，建議後續進行更細緻的理論詮釋與模型優化。
- 詳細分布圖與特徵相關性請見 results/figures/ 目錄。
