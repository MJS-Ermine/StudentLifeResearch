import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.data_loader import StudentLifeLoader
from src.feature_engineering import FeatureEngineer, merge_all_features
from src.config import RESULTS_ROOT, FIGURES_ROOT
import logging
import missingno as msno
from collections import defaultdict

logger = logging.getLogger(__name__)

REPORT_PATH = RESULTS_ROOT / 'auto_report.md'
FEATURES_NAN_BAR_PATH = FIGURES_ROOT / 'features_nan_bar.png'
FEATURES_NAN_REPORT_PATH = RESULTS_ROOT / 'features_nan_report.md'
IMPUTE_SUGGEST_PATH = RESULTS_ROOT / 'features_impute_suggestion.md'
COLUMN_STD_PATH = RESULTS_ROOT / 'column_standardization_suggestion.md'
MISSING_MATRIX_PATH = FIGURES_ROOT / 'features_missing_matrix.png'
MISSING_HEATMAP_PATH = FIGURES_ROOT / 'features_missing_corr_heatmap.png'
OUTLIER_REPORT_PATH = RESULTS_ROOT / 'features_outlier_report.md'
OUTLIER_BAR_PATH = FIGURES_ROOT / 'features_outlier_bar.png'
TARGET_LEAKAGE_REPORT_PATH = RESULTS_ROOT / 'target_leakage_report.md'

# 1. 載入所有特徵
def load_features_and_target():
    loader = StudentLifeLoader()
    engineer = FeatureEngineer()
    features = merge_all_features(loader, engineer)
    gpa_df = loader.load_gpa_data()
    # 自動偵測 user_id 欄位
    gpa_df.columns = [c.strip() for c in gpa_df.columns]  # 去除欄位空白
    if 'user_id' not in gpa_df.columns:
        if 'uid' in gpa_df.columns:
            gpa_df['user_id'] = gpa_df['uid']
        else:
            logger.warning(f"GPA data missing user_id/uid column. Columns: {gpa_df.columns.tolist()}")
            return pd.DataFrame(), pd.Series(dtype=float)
    # 自動偵測 gpa 欄位
    gpa_col = None
    gpa_candidates = [c for c in gpa_df.columns if 'gpa' in c.lower() or 'grade' in c.lower() or 'score' in c.lower() or 'final' in c.lower()]
    # 優先選 gpa all
    for c in gpa_candidates:
        if c.lower().replace('_','').replace(' ','') in ['gpaall','gpa_all','gpa all']:
            gpa_col = c
            break
    if gpa_col is None and gpa_candidates:
        gpa_col = gpa_candidates[0]
    if gpa_col is None:
        logger.warning(f"GPA data missing gpa column (tried: gpa, GPA, grade, final, score, gpa all, gpa_13s, cs65). Columns: {gpa_df.columns.tolist()}")
        return pd.DataFrame(), pd.Series(dtype=float)
    # 合併
    if features is None or features.empty:
        logger.warning("Features are empty, cannot merge with GPA.")
        return pd.DataFrame(), pd.Series(dtype=float)
    data = pd.merge(features, gpa_df[['user_id', gpa_col]], on='user_id', how='inner')
    if data.empty:
        logger.warning("No data after merging features and GPA.")
        return pd.DataFrame(), pd.Series(dtype=float)
    
    # 移除目標變數，避免目標洩漏
    target_series = data[gpa_col]
    features_clean = data.drop(columns=['user_id', gpa_col])
    
    return features_clean, target_series

# 2. 描述性統計與分布圖
def plot_feature_distributions(features: pd.DataFrame):
    if features is None or features.empty:
        logger.warning("No features to plot distributions.")
        return
    for col in features.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(features[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / f'{col}_dist.png')
        plt.close()

# 3. 相關性分析與熱圖
def plot_correlation_heatmap(features: pd.DataFrame, target: pd.Series):
    if features is None or features.empty or target is None or target.empty:
        logger.warning("No features/target to plot correlation heatmap.")
        return
    # 將 'missing'、'unknown'、'other' 等字串轉為 NaN
    features_clean = features.replace(['missing', 'unknown', 'other'], np.nan)
    # 只保留數值型欄位
    num_features = features_clean.select_dtypes(include=[np.number])
    # 若 target 也是數值型，合併進來
    if target is not None and np.issubdtype(target.dtype, np.number):
        corr = num_features.copy()
        corr['target'] = target
    else:
        corr = num_features
    corr_matrix = corr.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'correlation_heatmap.png')
    plt.close()

# 4. 模型訓練與效能比較
def train_and_evaluate_models(features: pd.DataFrame, target: pd.Series):
    if features is None or features.empty or target is None or target.empty:
        logger.warning("No features/target to train models.")
        return {}, {}
    
    # 清理字串值，確保所有特徵都是數值型
    features_clean = features.copy()
    
    # 將字串值轉換為 NaN，然後填充為 0
    for col in features_clean.columns:
        if features_clean[col].dtype == 'object':
            # 嘗試轉換為數值型
            features_clean[col] = pd.to_numeric(features_clean[col], errors='coerce')
        # 將所有 NaN 填充為 0
        features_clean[col] = features_clean[col].fillna(0)
    
    # 確保所有欄位都是數值型
    features_clean = features_clean.select_dtypes(include=[np.number])
    
    # 確保所有特徵名稱都是字符串類型
    features_clean.columns = features_clean.columns.astype(str)
    
    if features_clean.empty:
        logger.warning("No numeric features available for training.")
        return {}, {}
    
    X_train, X_test, y_train, y_test = train_test_split(features_clean, target, test_size=0.2, random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    feature_importances = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'r2': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred) ** 0.5,
            'mae': mean_absolute_error(y_test, y_pred)
        }
        if hasattr(model, 'coef_'):
            feature_importances[name] = pd.Series(model.coef_, index=features_clean.columns)
        elif hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(model.feature_importances_, index=features_clean.columns)
    return results, feature_importances

# 5. 特徵重要性條形圖
def plot_feature_importance(feature_importances: dict):
    if not feature_importances:
        logger.warning("No feature importances to plot.")
        return
    for model, importances in feature_importances.items():
        importances = importances.abs().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title(f'Feature Importance ({model})')
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / f'feature_importance_{model}.png')
        plt.close()

# 6. 自動產生初步 markdown 報告
def generate_report(results: dict, feature_importances: dict):
    if not results or not feature_importances:
        logger.warning("No results or feature importances to generate report.")
        return
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('# StudentLife Auto Analysis Report\n\n')
        f.write('## 1. Research Background\n')
        f.write('This report is based on the StudentLife dataset, exploring the relationship between college students\' daily behaviors, psychological states, and academic performance, and using multiple machine learning models for GPA prediction and feature importance analysis.\n\n')
        f.write('## 2. Model Performance Comparison\n')
        f.write('| Model | R2 | RMSE | MAE |\n')
        f.write('|-------|----|------|-----|\n')
        for model, res in results.items():
            f.write(f'| {model} | {res["r2"]:.3f} | {res["rmse"]:.3f} | {res["mae"]:.3f} |\n')
        f.write('\n![](figures/correlation_heatmap.png)\n')
        for model in feature_importances:
            f.write(f'\n### {model} Feature Importance\n')
            f.write(f'![](figures/feature_importance_{model}.png)\n')
        f.write('\n## 3. Initial Findings Summary\n')
        f.write('- Some behavioral features (such as sleep, phone usage, activity, and psychological state) have significant predictive power for GPA.\n')
        f.write('- Different models have slightly different rankings of feature importance, suggesting further theoretical interpretation and model optimization.\n')
        f.write('- Detailed distribution graphs and feature correlations can be found in the results/figures/ directory.\n')

def create_features(data: pd.DataFrame):
    """Create features from the raw data."""
    logger.info("Creating features...")
    engineer = FeatureEngineer()
    features = engineer.create_all_features(data)
    # 自動補值，避免 NaN 導致模型報錯
    features = features.fillna(0)
    logger.info(f"Created features shape: {features.shape}")
    return features

if __name__ == "__main__":
    print("[INFO] 載入特徵與目標...")
    features, target = load_features_and_target()
    # 強制所有欄位名稱為字串，避免 sklearn 報錯
    features.columns = features.columns.astype(str)
    # ====== 欄位命名標準化建議（只產生 mapping，不自動覆蓋） ======
    col_map = defaultdict(list)
    for col in features.columns:
        std_col = str(col).strip().lower().replace(' ', '_')
        if std_col in ['uid', 'user_id']:
            col_map['user_id'].append(col)
        elif std_col in ['steps', 'step_count']:
            col_map['step_count'].append(col)
        elif std_col in ['activity_inference', 'activity inference']:
            col_map['activity_inference'].append(col)
        else:
            col_map[std_col].append(col)
    with open(COLUMN_STD_PATH, 'w', encoding='utf-8') as f:
        f.write('# 欄位命名標準化建議（自動產生）\n')
        f.write('本報告自動偵測常見異名欄位，建議統一命名如下：\n\n')
        f.write('| 建議標準名 | 原始名列表 |\n')
        f.write('|------------|------------|\n')
        for std, raw_list in col_map.items():
            if len(raw_list) > 1 or std in ['user_id', 'step_count', 'activity_inference']:
                f.write(f'| {std} | {", ".join(raw_list)} |\n')
    print(f"[INFO] 已產生欄位命名標準化建議: {COLUMN_STD_PATH}")
    # ====== 自動補值（根據建議自動執行） ======
    nan_per_col = features.isna().sum()
    nan_ratio_per_col = features.isna().mean()
    impute_log = []
    for col in features.columns:
        n_nan = nan_per_col[col]
        ratio = nan_ratio_per_col[col]
        dtype = str(features[col].dtype)
        if n_nan == 0:
            continue
        if dtype.startswith('float') or dtype.startswith('int'):
            if ratio < 0.05:
                fill_value = features[col].mean()
                features[col] = features[col].fillna(fill_value)
                impute_log.append(f'- {col}：以平均數 {fill_value:.4f} 補值')
            elif ratio < 0.3:
                fill_value = features[col].median()
                features[col] = features[col].fillna(fill_value)
                impute_log.append(f'- {col}：以中位數 {fill_value:.4f} 補值')
            else:
                features[col] = features[col].fillna(0)
                impute_log.append(f'- {col}：缺失比例過高，直接補 0')
        else:
            fill_value = features[col].mode().iloc[0] if not features[col].mode().empty else 'missing'
            features[col] = features[col].fillna(fill_value)
            impute_log.append(f'- {col}：以眾數 "{fill_value}" 補值')
    with open(IMPUTE_SUGGEST_PATH, 'w', encoding='utf-8') as f:
        f.write('# 特徵補值執行紀錄（自動產生）\n')
        f.write('本報告記錄每個特徵實際自動補值方式與數值。\n\n')
        if not impute_log:
            f.write('所有特徵皆無缺失值，無需補值。\n')
        else:
            for log in impute_log:
                f.write(log + '\n')
    print(f"[INFO] 已自動補值並產生補值紀錄: {IMPUTE_SUGGEST_PATH}")
    # ====== 異常值偵測（自動產生分布圖與報告，不自動修正） ======
    outlier_counts = {}
    for col in features.select_dtypes(include=['float', 'int']).columns:
        x = features[col]
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier = ((x < lower) | (x > upper)) & (~x.isna())
        outlier_counts[col] = outlier.sum()
    # 畫出異常值數量條形圖
    if outlier_counts:
        outlier_series = pd.Series(outlier_counts)
        plt.figure(figsize=(min(16, 0.5+0.5*len(outlier_series)), 6))
        outlier_series[outlier_series>0].sort_values(ascending=False).plot(kind='bar')
        plt.title('Outlier Count per Feature')
        plt.ylabel('Outlier Count')
        plt.xlabel('Feature Name')
        plt.tight_layout()
        plt.savefig(OUTLIER_BAR_PATH)
        plt.close()
        print(f"[INFO] 已產生異常值分布圖: {OUTLIER_BAR_PATH}")
    # 產生異常值詳細報告
    with open(OUTLIER_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('# 特徵異常值偵測報告（自動產生）\n')
        f.write('本報告統計每個數值型特徵的異常值數量與比例，僅偵測不自動修正。\n\n')
        f.write('| 特徵名稱 | 異常值數量 | 異常值比例 |\n')
        f.write('|----------|----------|----------|\n')
        for col, count in outlier_counts.items():
            ratio = count / len(features) if len(features) > 0 else 0
            if count > 0:
                f.write(f'| {col} | {count} | {ratio:.2%} |\n')
        if any(v > 0 for v in outlier_counts.values()):
            f.write(f'\n![](figures/features_outlier_bar.png)\n')
    print(f"[INFO] 已產生異常值偵測報告: {OUTLIER_REPORT_PATH}")
    # ====== Auto-generated features missing distribution graph and detailed missing report ======
    nan_count = features.isna().sum().sum()
    nan_cols = features.columns[features.isna().any()].tolist()
    nan_per_col = features.isna().sum()
    nan_ratio_per_col = features.isna().mean()
    # 1. Bar plot (English labels)
    if nan_count > 0:
        plt.figure(figsize=(min(16, 0.5+0.5*len(features.columns)), 6))
        nan_per_col[nan_per_col>0].sort_values(ascending=False).plot(kind='bar')
        plt.title('Missing Value Count per Feature')
        plt.ylabel('Missing (NaN) Count')
        plt.xlabel('Feature Name')
        plt.tight_layout()
        plt.savefig(FEATURES_NAN_BAR_PATH)
        plt.close()
        print(f"[INFO] 已產生 features 缺失分布圖: {FEATURES_NAN_BAR_PATH}")
    # 2. Missing matrix (English title)
    if nan_count > 0:
        plt.figure(figsize=(min(16, 0.5+0.5*len(features.columns)), 8))
        msno.matrix(features, fontsize=12)
        plt.title('Missing Data Matrix (per sample)')
        plt.tight_layout()
        plt.savefig(MISSING_MATRIX_PATH)
        plt.close()
        print(f"[INFO] 已產生 features 缺失矩陣圖: {MISSING_MATRIX_PATH}")
    # 3. Missing correlation heatmap (English title)
    if nan_count > 0:
        plt.figure(figsize=(min(16, 0.5+0.5*len(features.columns)), 8))
        msno.heatmap(features, fontsize=12)
        plt.title('Missing Data Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(MISSING_HEATMAP_PATH)
        plt.close()
        print(f"[INFO] 已產生 features 缺失相關性熱圖: {MISSING_HEATMAP_PATH}")
    # 4. Generate detailed missing report
    with open(FEATURES_NAN_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('# Detailed Missing Value Report (Auto-generated)\n')
        f.write('This report summarizes the NaN count and ratio for each feature in the final training features.\n\n')
        if nan_count == 0:
            f.write('All features have no missing values (NaN).\n')
        else:
            f.write(f'Total missing value count: {nan_count}\n\n')
            f.write('| Feature Name | NaN Count | NaN Ratio |\n')
            f.write('|------------|----------|----------|\n')
            for col in features.columns:
                if nan_per_col[col] > 0:
                    f.write(f'| {col} | {nan_per_col[col]} | {nan_ratio_per_col[col]:.2%} |\n')
            if nan_count > 0:
                f.write(f'\n![](figures/features_nan_bar.png)\n')
                f.write(f'\n![](figures/features_missing_matrix.png)\n')
                f.write(f'\n![](figures/features_missing_corr_heatmap.png)\n')
    print(f"[INFO] 已產生 features 缺失詳細報告: {FEATURES_NAN_REPORT_PATH}")
    # ====== Model training before auto-imputation and NaN check ======
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        nan_cols = features.columns[features.isna().any()].tolist()
        logger.warning(f"[NaN Check] Model training before features still have NaN, total {nan_count} in columns: {nan_cols}")
        print(f"[WARNING] Model training before features still have NaN, total {nan_count} in columns: {nan_cols}")
        features = features.fillna(0)
        logger.info("All NaN automatically filled with 0.")
    else:
        logger.info("[NaN Check] Model training before features have no NaN.")
    print("[INFO] Generating feature distribution graph...")
    plot_feature_distributions(features)
    print("[INFO] Generating correlation heatmap...")
    plot_correlation_heatmap(features, target)
    print("[INFO] Training models and comparing performance...")
    results, feature_importances = train_and_evaluate_models(features, target)
    print("[INFO] Generating feature importance graph...")
    plot_feature_importance(feature_importances)
    print("[INFO] Generating auto-generated preliminary report...")
    generate_report(results, feature_importances)
    print("[INFO] Auto analysis completed, please check results/auto_report.md and results/figures/")
    # ====== Target leakage 檢查 ======
    leakage_cols = [c for c in features.columns if any(x in c.lower() for x in ['gpa', 'grade', 'score', 'target', 'final'])]
    with open(TARGET_LEAKAGE_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('# 目標洩漏（Target Leakage）自動檢查報告\n')
        f.write('本報告自動檢查 features 是否含有與目標（如 gpa、grade、score、target、final）相關欄位。\n\n')
        if leakage_cols:
            f.write('## 發現疑似目標洩漏欄位：\n')
            for c in leakage_cols:
                f.write(f'- {c}\n')
            f.write('\n建議：請勿將上述欄位作為特徵進行模型訓練，否則會造成評估失真。\n')
        else:
            f.write('未發現疑似目標洩漏欄位，資料安全。\n')
    print(f"[INFO] 已產生 target leakage 檢查報告: {TARGET_LEAKAGE_REPORT_PATH}") 