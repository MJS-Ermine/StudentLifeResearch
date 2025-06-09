# StudentLifeResearch

## 主要功能
- 自動化資料檢查、特徵工程、模型訓練、報告產生
- 支援多種感測器、問卷、EMA 資料
- 數據品質稽核、缺失/異常/洩漏檢查

## 進階功能（裝逼專用）
### 1. 互動式 EDA 儀表板
- `python -m src.eda.eda_dashboard`
- Streamlit 互動式資料總覽、分布、相關性、缺失、異常值分析

### 2. 自動化模型超參數搜尋（AutoML）
- `python -m src.automl.automl_search`
- Optuna 自動搜尋最佳模型參數，結果儲存於 `results/automl_best_params.txt`

### 3. 模型解釋性（Explainable AI, XAI）
- `python -m src.xai.shap_explain`
- SHAP summary plot、force plot 自動產生於 `results/`

## 安裝依賴
```bash
pip install -r requirements.txt
```

## 資料夾結構
- `src/eda/eda_dashboard.py`：EDA 儀表板
- `src/automl/automl_search.py`：AutoML 超參數搜尋
- `src/xai/shap_explain.py`：SHAP 模型解釋

## 注意事項
- 請先準備好 data/dataset 目錄下的原始資料
- 若遇到缺失或錯誤，請先執行 `python src/check_dataset.py` 檢查
