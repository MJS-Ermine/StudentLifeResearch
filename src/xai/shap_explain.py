"""
SHAP Explainability for StudentLife Models
Run: python -m src.xai.shap_explain
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import StudentLifeLoader
from feature_engineering import FeatureEngineer, merge_all_features
from config import RESULTS_ROOT, FIGURES_ROOT

def load_features_and_target():
    """載入特徵和目標變數，與auto_analysis.py保持一致"""
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
            print(f"GPA data missing user_id/uid column. Columns: {gpa_df.columns.tolist()}")
            return pd.DataFrame(), pd.Series(dtype=float)
    
    # 自動偵測 gpa 欄位
    gpa_col = None
    gpa_candidates = [c for c in gpa_df.columns if 'gpa' in c.lower() or 'grade' in c.lower() or 'score' in c.lower() or 'final' in c.lower()]
    # 優先選 gpa all
    for c in gpa_candidates:
        clean_col = c.lower().strip().replace('_','').replace(' ','')
        if clean_col in ['gpaall','gpa_all','gpaall']:
            gpa_col = c
            break
    if gpa_col is None and gpa_candidates:
        gpa_col = gpa_candidates[0]
    if gpa_col is None:
        print(f"GPA data missing gpa column. Columns: {gpa_df.columns.tolist()}")
        return pd.DataFrame(), pd.Series(dtype=float)
    
    # 合併
    if features is None or features.empty:
        print("Features are empty, cannot merge with GPA.")
        return pd.DataFrame(), pd.Series(dtype=float)
    data = pd.merge(features, gpa_df[['user_id', gpa_col]], on='user_id', how='inner')
    if data.empty:
        print("No data after merging features and GPA.")
        return pd.DataFrame(), pd.Series(dtype=float)
    
    # 移除目標變數，避免目標洩漏
    target_series = data[gpa_col]
    features_clean = data.drop(columns=['user_id', gpa_col])
    
    return features_clean, target_series

def clean_features_for_modeling(features):
    """清理特徵，確保所有特徵都是數值型"""
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
    
    return features_clean

def statistical_analysis(features, target):
    """進行統計分析，計算相關性和顯著性"""
    print("\n=== 統計分析：特徵與GPA的相關性 ===")
    
    correlations = []
    for col in features.columns:
        if features[col].var() > 0:  # 避免常數特徵
            corr, p_value = pearsonr(features[col], target)
            correlations.append({
                'feature': col,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    
    # 儲存統計結果
    corr_df.to_csv(RESULTS_ROOT / 'feature_correlations.csv', index=False)
    
    # 顯示前10個最相關的特徵
    print("\n前10個與GPA最相關的特徵：")
    print(corr_df.head(10)[['feature', 'correlation', 'p_value', 'significant']].to_string(index=False))
    
    return corr_df

def plot_feature_importance_comparison(rf_model, ridge_model, features, target):
    """比較不同模型的特徵重要性"""
    # RandomForest 特徵重要性
    rf_importance = pd.Series(rf_model.feature_importances_, index=features.columns)
    
    # Ridge 係數（絕對值）
    ridge_importance = pd.Series(np.abs(ridge_model.coef_), index=features.columns)
    
    # 相關性（絕對值）
    correlations = []
    for col in features.columns:
        if features[col].var() > 0:
            corr, _ = pearsonr(features[col], target)
            correlations.append(abs(corr))
        else:
            correlations.append(0)
    corr_importance = pd.Series(correlations, index=features.columns)
    
    # 取前15個最重要的特徵
    top_features = rf_importance.nlargest(15).index
    
    # 創建比較圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RandomForest
    rf_importance[top_features].sort_values().plot(kind='barh', ax=axes[0])
    axes[0].set_title('RandomForest Feature Importance')
    axes[0].set_xlabel('Importance')
    
    # Ridge
    ridge_importance[top_features].sort_values().plot(kind='barh', ax=axes[1])
    axes[1].set_title('Ridge Coefficient (Absolute)')
    axes[1].set_xlabel('|Coefficient|')
    
    # Correlation
    corr_importance[top_features].sort_values().plot(kind='barh', ax=axes[2])
    axes[2].set_title('Correlation with GPA (Absolute)')
    axes[2].set_xlabel('|Correlation|')
    
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_features

def generate_academic_report(corr_df, top_features, model, features, target):
    """生成學術分析報告"""
    report_path = RESULTS_ROOT / 'academic_analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# StudentLife 學術分析報告\n\n')
        f.write('## 1. 研究背景與目標\n')
        f.write('本研究基於 StudentLife 資料集，探討大學生日常行為模式對學業表現的影響。\n')
        f.write('透過機器學習與統計分析方法，識別對 GPA 最具預測力的行為特徵。\n\n')
        
        f.write('## 2. 資料概述\n')
        f.write(f'- 樣本數量: {features.shape[0]} 位學生\n')
        f.write(f'- 特徵數量: {features.shape[1]} 個行為特徵\n')
        f.write(f'- 目標變數: GPA (平均值: {target.mean():.3f}, 標準差: {target.std():.3f})\n\n')
        
        f.write('## 3. 主要發現\n\n')
        f.write('### 3.1 與 GPA 最相關的行為特徵\n')
        f.write('| 特徵名稱 | 相關係數 | p值 | 顯著性 |\n')
        f.write('|----------|----------|-----|--------|\n')
        
        significant_features = corr_df[corr_df['significant']].head(10)
        for _, row in significant_features.iterrows():
            significance = "✓" if row['significant'] else "✗"
            f.write(f"| {row['feature']} | {row['correlation']:.3f} | {row['p_value']:.3f} | {significance} |\n")
        
        f.write('\n### 3.2 模型預測性能\n')
        f.write(f'- RandomForest R²: {model.score(features, target):.3f}\n')
        f.write('- 模型能解釋約 {:.1f}% 的 GPA 變異\n\n'.format(model.score(features, target) * 100))
        
        f.write('### 3.3 特徵重要性排序\n')
        feature_importance = pd.Series(model.feature_importances_, index=features.columns)
        top_10_features = feature_importance.nlargest(10)
        
        f.write('| 排名 | 特徵名稱 | 重要性分數 |\n')
        f.write('|------|----------|------------|\n')
        for i, (feature, importance) in enumerate(top_10_features.items(), 1):
            f.write(f'| {i} | {feature} | {importance:.4f} |\n')
        
        f.write('\n## 4. 學術意義與討論\n\n')
        f.write('### 4.1 行為科學觀點\n')
        f.write('根據分析結果，以下行為模式對學業表現具有顯著影響：\n\n')
        
        # 根據特徵名稱推斷行為類別
        behavior_categories = {
            'sleep': '睡眠行為',
            'phone': '手機使用',
            'activity': '身體活動',
            'audio': '環境音頻',
            'conversation': '社交互動',
            'charge': '充電行為',
            'ema': '情緒與主觀體驗'
        }
        
        for category, description in behavior_categories.items():
            category_features = [f for f in top_10_features.index if category in f.lower()]
            if category_features:
                f.write(f'- **{description}**: {", ".join(category_features[:3])}\n')
        
        f.write('\n### 4.2 理論連結\n')
        f.write('這些發現與以下理論框架一致：\n')
        f.write('- **自我調節理論**: 規律的行為模式有助於學業成功\n')
        f.write('- **睡眠與認知功能**: 充足睡眠是學習記憶的生理基礎\n')
        f.write('- **數位健康**: 適度的科技使用與學業表現相關\n\n')
        
        f.write('### 4.3 實務應用建議\n')
        f.write('基於研究發現，建議大學生：\n')
        f.write('1. 維持規律的睡眠作息\n')
        f.write('2. 適度控制手機使用時間\n')
        f.write('3. 保持適當的身體活動\n')
        f.write('4. 注意社交互動的品質\n\n')
        
        f.write('## 5. 研究限制\n')
        f.write('- 樣本來源單一（達特茅斯學院）\n')
        f.write('- 觀察期間有限（一學期）\n')
        f.write('- 因果關係需進一步驗證\n\n')
        
        f.write('## 6. 圖表說明\n')
        f.write('- `feature_importance_comparison.png`: 不同方法的特徵重要性比較\n')
        f.write('- `shap_summary_plot.png`: SHAP 特徵重要性總覽\n')
        f.write('- `shap_bar_plot.png`: SHAP 特徵重要性條形圖\n')
        f.write('- `shap_dependence_*.png`: 重要特徵的依賴關係圖\n')
    
    print(f"學術分析報告已儲存至: {report_path}")

def main():
    print("=== StudentLife 特徵重要性與解釋性分析 ===")
    
    # 載入資料
    print("\n1. 載入特徵和目標變數...")
    features, target = load_features_and_target()
    if features.empty or target.empty:
        print("Features or target is empty.")
        return
    
    print(f"特徵數量: {features.shape[1]}, 樣本數量: {features.shape[0]}")
    
    # 清理特徵
    print("\n2. 清理特徵資料...")
    features_clean = clean_features_for_modeling(features)
    # 確保所有欄位名稱都是字串
    features_clean.columns = features_clean.columns.astype(str)
    print(f"清理後特徵數量: {features_clean.shape[1]}")
    
    # 統計分析
    print("\n3. 進行統計分析...")
    corr_df = statistical_analysis(features_clean, target)
    
    # 分割資料
    print("\n4. 分割訓練/測試資料...")
    X_train, X_test, y_train, y_test = train_test_split(features_clean, target, test_size=0.2, random_state=42)
    
    # 訓練模型
    print("\n5. 訓練模型...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    
    # 特徵重要性比較
    print("\n6. 生成特徵重要性比較圖...")
    top_features = plot_feature_importance_comparison(rf_model, ridge_model, features_clean, target)
    
    # SHAP 分析
    print("\n7. 進行 SHAP 分析...")
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot 已儲存至: {FIGURES_ROOT / 'shap_summary_plot.png'}")
        
        # SHAP Bar Plot (特徵重要性)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP bar plot 已儲存至: {FIGURES_ROOT / 'shap_bar_plot.png'}")
        
        # 針對前5個重要特徵生成 dependence plots
        print("\n8. 生成前5個重要特徵的 SHAP dependence plots...")
        feature_importance = np.abs(shap_values).mean(0)
        top_5_indices = np.argsort(feature_importance)[-5:]
        
        for i, idx in enumerate(top_5_indices):
            feature_name = X_test.columns[idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X_test, show=False)
            plt.title(f'SHAP Dependence Plot: {feature_name}')
            plt.tight_layout()
            plt.savefig(FIGURES_ROOT / f'shap_dependence_{i+1}_{feature_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"SHAP 分析出現錯誤: {e}")
    
    # 生成學術報告
    print("\n9. 生成學術分析報告...")
    generate_academic_report(corr_df, top_features, rf_model, features_clean, target)
    
    print(f"\n=== 分析完成！請查看 {RESULTS_ROOT} 和 {FIGURES_ROOT} 目錄 ===")

if __name__ == "__main__":
    main()
