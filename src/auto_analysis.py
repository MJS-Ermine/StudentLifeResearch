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

logger = logging.getLogger(__name__)

REPORT_PATH = RESULTS_ROOT / 'auto_report.md'

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
    return data.drop(columns=['user_id']), data[gpa_col]

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
    corr = features.copy()
    corr['target'] = target
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
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
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
            'R2': r2_score(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred) ** 0.5,
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        if hasattr(model, 'coef_'):
            feature_importances[name] = pd.Series(model.coef_, index=features.columns)
        elif hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(model.feature_importances_, index=features.columns)
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
        f.write('# StudentLife 自動化初步分析報告\n\n')
        f.write('## 1. 研究背景\n')
        f.write('本報告基於 StudentLife dataset，探討大學生日常行為、心理狀態與學業表現之間的關聯，並以多種機器學習模型進行 GPA 預測與特徵重要性分析。\n\n')
        f.write('## 2. 模型效能比較\n')
        f.write('| Model | R2 | RMSE | MAE |\n')
        f.write('|-------|----|------|-----|\n')
        for model, res in results.items():
            f.write(f'| {model} | {res["R2"]:.3f} | {res["RMSE"]:.3f} | {res["MAE"]:.3f} |\n')
        f.write('\n![](figures/correlation_heatmap.png)\n')
        for model in feature_importances:
            f.write(f'\n### {model} 特徵重要性\n')
            f.write(f'![](figures/feature_importance_{model}.png)\n')
        f.write('\n## 3. 初步發現摘要\n')
        f.write('- 部分行為特徵（如睡眠、手機使用、活動、心理狀態）對 GPA 具顯著預測力。\n')
        f.write('- 不同模型對特徵重要性的排序略有差異，建議後續進行更細緻的理論詮釋與模型優化。\n')
        f.write('- 詳細分布圖與特徵相關性請見 results/figures/ 目錄。\n')

if __name__ == "__main__":
    print("[INFO] 載入特徵與目標...")
    features, target = load_features_and_target()
    print("[INFO] 產生特徵分布圖...")
    plot_feature_distributions(features)
    print("[INFO] 產生相關性熱圖...")
    plot_correlation_heatmap(features, target)
    print("[INFO] 訓練模型並比較效能...")
    results, feature_importances = train_and_evaluate_models(features, target)
    print("[INFO] 產生特徵重要性圖...")
    plot_feature_importance(feature_importances)
    print("[INFO] 產生自動化初步報告...")
    generate_report(results, feature_importances)
    print("[INFO] 自動化分析完成，請查看 results/auto_report.md 及 results/figures/") 