"""
AutoML Hyperparameter Search for StudentLife Dataset
Run: python -m src.automl.automl_search
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import StudentLifeLoader
from feature_engineering import FeatureEngineer
from config import RESULTS_ROOT

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

def load_survey_features_and_target():
    from glob import glob
    import os

    project_root = Path(__file__).resolve().parents[2]
    survey_dir = project_root / "data" / "dataset" / "survey"
    print("survey_dir:", survey_dir)
    # 讀取所有 csv
    survey_files = list(survey_dir.glob("*.csv"))
    print("survey_files:", survey_files)
    dfs = []
    for f in survey_files:
        print("Reading:", f)
        df = pd.read_csv(f)
        print("Columns:", df.columns)
        # 標準化 user_id 欄位
        if 'user_id' not in df.columns and 'uid' in df.columns:
            df = df.rename(columns={'uid': 'user_id'})
        if 'user_id' in df.columns:
            # 只保留數值型欄位
            num_cols = df.select_dtypes(include=[np.number]).columns
            df_grouped = df.groupby('user_id')[num_cols].mean()
            dfs.append(df_grouped)
        else:
            print(f"[警告] {f} 沒有 user_id 或 uid 欄位，已跳過")
    # 合併所有問卷
    features = pd.concat(dfs, axis=1)
    # 讀取 GPA
    grades = pd.read_csv(survey_dir.parent / "education" / "grades.csv")
    if 'user_id' not in grades.columns and 'uid' in grades.columns:
        grades = grades.rename(columns={'uid': 'user_id'})
    gpa_col = [c for c in grades.columns if 'gpa' in c.lower() or 'score' in c.lower() or 'grade' in c.lower()][0]
    target = grades.set_index('user_id')[gpa_col]
    # 只保留有 target 的樣本
    features = features.loc[features.index.intersection(target.index)]
    target = target.loc[features.index]
    features = features.fillna(features.mean())
    return features, target

def objective(trial):
    features, target = load_survey_features_and_target()
    if features.empty or target.empty:
        return float('inf')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features)
    X_test = scaler.transform(features)
    model_name = trial.suggest_categorical('model', [
        'RandomForest', 'Ridge', 'SVR', 'KNN', 'MLP',
        'XGBoost' if XGBRegressor else None,
        'LightGBM' if LGBMRegressor else None
    ])
    if model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    elif model_name == 'Ridge':
        alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
        model = Ridge(alpha=alpha, random_state=42)
    elif model_name == 'SVR':
        c = trial.suggest_float('C', 0.1, 10.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
        model = SVR(C=c, epsilon=epsilon)
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_name == 'MLP':
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50,50)])
        alpha = trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=300, random_state=42)
    elif model_name == 'XGBoost' and XGBRegressor:
        n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 20)
        learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbosity=0)
    elif model_name == 'LightGBM' and LGBMRegressor:
        n_estimators = trial.suggest_int('lgb_n_estimators', 50, 300)
        max_depth = trial.suggest_int('lgb_max_depth', 3, 20)
        learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
        model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)
    else:
        return float('inf')
    model.fit(X_train, target)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(target, y_pred) ** 0.5
    return rmse

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print("Best trial:")
    print(study.best_trial)
    print("Best model:", study.best_trial.params)
    # 儲存最佳參數與結果
    with open(RESULTS_ROOT / 'automl_best_params.txt', 'w', encoding='utf-8') as f:
        f.write(str(study.best_trial))
    print(f"Best params saved to {RESULTS_ROOT / 'automl_best_params.txt'}")

    # --- 畫圖 ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    trials_df = study.trials_dataframe()
    # 過濾掉 model=None 的 trial
    trials_df = trials_df[trials_df['params_model'].notnull()]
    # 1. RMSE vs. trial
    plt.figure(figsize=(8,4))
    plt.plot(trials_df['number'], trials_df['value'], marker='o')
    plt.xlabel('Trial')
    plt.ylabel('RMSE')
    plt.title('AutoML Trials RMSE')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'automl_trials_rmse.png')
    # 2. 各模型最佳RMSE
    best_per_model = trials_df.groupby('params_model')['value'].min().sort_values()
    plt.figure(figsize=(8,4))
    sns.barplot(x=best_per_model.index, y=best_per_model.values)
    plt.ylabel('Best RMSE')
    plt.xlabel('Model')
    plt.title('Best RMSE by Model')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'automl_best_rmse_by_model.png')
    # 3. RMSE vs. trial by model
    plt.figure(figsize=(10,5))
    for model in trials_df['params_model'].unique():
        sub = trials_df[trials_df['params_model'] == model]
        plt.plot(sub['number'], sub['value'], marker='o', label=model)
    plt.xlabel('Trial')
    plt.ylabel('RMSE')
    plt.title('AutoML Trials RMSE by Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'automl_trials_rmse_by_model.png')
    print(f"圖表已儲存於 {RESULTS_ROOT}")

    # 4. 用最佳模型重訓並產生預測結果
    best_params = study.best_trial.params
    best_model_name = best_params['model']
    features, target = load_survey_features_and_target()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    # 依照最佳模型名稱與參數建立模型
    if best_model_name == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            random_state=42
        )
    elif best_model_name == 'Ridge':
        model = Ridge(alpha=best_params.get('alpha', 1.0), random_state=42)
    elif best_model_name == 'SVR':
        model = SVR(C=best_params.get('C', 1.0), epsilon=best_params.get('epsilon', 0.1))
    elif best_model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors=best_params.get('n_neighbors', 5))
    elif best_model_name == 'MLP':
        model = MLPRegressor(
            hidden_layer_sizes=best_params.get('hidden_layer_sizes', (100,)),
            alpha=best_params.get('mlp_alpha', 0.0001),
            max_iter=300,
            random_state=42
        )
    elif best_model_name == 'XGBoost' and XGBRegressor:
        model = XGBRegressor(
            n_estimators=best_params.get('xgb_n_estimators', 100),
            max_depth=best_params.get('xgb_max_depth', 3),
            learning_rate=best_params.get('xgb_learning_rate', 0.1),
            random_state=42,
            verbosity=0
        )
    elif best_model_name == 'LightGBM' and LGBMRegressor:
        model = LGBMRegressor(
            n_estimators=best_params.get('lgb_n_estimators', 100),
            max_depth=best_params.get('lgb_max_depth', 3),
            learning_rate=best_params.get('lgb_learning_rate', 0.1),
            random_state=42,
            verbose=-1
        )
    else:
        print("Best model not recognized.")
        return

    # 切分資料
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 計算指標
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"Best Model Test RMSE: {rmse:.4f}")
    print(f"Best Model Test R2: {r2:.4f}")

    # 預測值 vs. 真實值散點圖（測試集）
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('True GPA')
    plt.ylabel('Predicted GPA')
    plt.title('Best Model: True vs. Predicted GPA (Test Set)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'best_model_true_vs_pred.png')

    # 殘差分布圖（測試集）
    plt.figure(figsize=(6,4))
    plt.hist(y_test - y_pred, bins=20)
    plt.xlabel('Residual (True - Predicted)')
    plt.ylabel('Count')
    plt.title('Best Model Residual Distribution (Test Set)')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'best_model_residuals.png')

    # 預測值與真實值對照表（測試集）
    pd.DataFrame({'True': y_test, 'Predicted': y_pred}).to_csv(RESULTS_ROOT / 'best_model_predictions.csv', index=False)

    # 預測值 vs. 真實值散點圖（全部資料）
    y_pred_all = model.predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(target, y_pred_all, alpha=0.7)
    plt.xlabel('True GPA')
    plt.ylabel('Predicted GPA')
    plt.title('Best Model: True vs. Predicted GPA (All Data)')
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'best_model_true_vs_pred_all.png')

    # 殘差分布圖（全部資料）
    plt.figure(figsize=(6,4))
    plt.hist(target - y_pred_all, bins=20)
    plt.xlabel('Residual (True - Predicted)')
    plt.ylabel('Count')
    plt.title('Best Model Residual Distribution (All Data)')
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'best_model_residuals_all.png')

    # 預測值與真實值對照表（全部資料）
    pd.DataFrame({'True': target, 'Predicted': y_pred_all}).to_csv(RESULTS_ROOT / 'best_model_predictions_all.csv', index=False)

    print(f"最佳模型結果圖與預測表（測試集+全部資料）已儲存於 {RESULTS_ROOT}")

if __name__ == "__main__":
    main()

