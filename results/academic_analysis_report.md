# StudentLife 學術分析報告

## 1. 研究背景與目標
本研究基於 StudentLife 資料集，探討大學生日常行為模式對學業表現的影響。
透過機器學習與統計分析方法，識別對 GPA 最具預測力的行為特徵。

## 2. 資料概述
- 樣本數量: 30 位學生
- 特徵數量: 30 個行為特徵
- 目標變數: GPA (平均值: 3.422, 標準差: 0.398)

## 3. 主要發現

### 3.1 與 GPA 最相關的行為特徵
| 特徵名稱 | 相關係數 | p值 | 顯著性 |
|----------|----------|-----|--------|
| activity_prop_2 | -0.510 | 0.004 | ✓ |
| activity_prop_0 | 0.408 | 0.025 | ✓ |
| night_charge_freq | 0.407 | 0.026 | ✓ |
| activity_entropy | -0.397 | 0.030 | ✓ |

### 3.2 模型預測性能
- RandomForest R²: 0.735
- 模型能解釋約 73.5% 的 GPA 變異

### 3.3 特徵重要性排序
| 排名 | 特徵名稱 | 重要性分數 |
|------|----------|------------|
| 1 | activity_onehot_0 | 0.2261 |
| 2 | night_charge_freq | 0.2236 |
| 3 | ema_pam_picture_idx_mean | 0.1213 |
| 4 | ema_pam_extreme_picture_idx | 0.1076 |
| 5 | activity_prop_2 | 0.1003 |
| 6 | ema_pam_picture_idx_std | 0.0547 |
| 7 | activity_prop_3 | 0.0280 |
| 8 | activity_onehot_2 | 0.0254 |
| 9 | ema_pam_unique_picture_idx | 0.0190 |
| 10 | activity_onehot_1 | 0.0184 |

## 4. 學術意義與討論

### 4.1 行為科學觀點
根據分析結果，以下行為模式對學業表現具有顯著影響：

- **身體活動**: activity_onehot_0, activity_prop_2, activity_prop_3
- **充電行為**: night_charge_freq
- **情緒與主觀體驗**: ema_pam_picture_idx_mean, ema_pam_extreme_picture_idx, ema_pam_picture_idx_std

### 4.2 理論連結
這些發現與以下理論框架一致：
- **自我調節理論**: 規律的行為模式有助於學業成功
- **睡眠與認知功能**: 充足睡眠是學習記憶的生理基礎
- **數位健康**: 適度的科技使用與學業表現相關

### 4.3 實務應用建議
基於研究發現，建議大學生：
1. 維持規律的睡眠作息
2. 適度控制手機使用時間
3. 保持適當的身體活動
4. 注意社交互動的品質

## 5. 研究限制
- 樣本來源單一（達特茅斯學院）
- 觀察期間有限（一學期）
- 因果關係需進一步驗證

## 6. 圖表說明
- `feature_importance_comparison.png`: 不同方法的特徵重要性比較
- `shap_summary_plot.png`: SHAP 特徵重要性總覽
- `shap_bar_plot.png`: SHAP 特徵重要性條形圖
- `shap_dependence_*.png`: 重要特徵的依賴關係圖
