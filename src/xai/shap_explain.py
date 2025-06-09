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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import StudentLifeLoader
from feature_engineering import FeatureEngineer
from config import RESULTS_ROOT

def load_features_and_target():
    loader = StudentLifeLoader()
    engineer = FeatureEngineer()
    features = engineer.create_all_features(loader.merge_all_data())
    gpa = loader.load_gpa_data()
    if 'user_id' in gpa.columns:
        gpa_col = [c for c in gpa.columns if 'gpa' in c.lower()][0]
        target = gpa.set_index('user_id')[gpa_col]
        features = features.set_index('user_id').loc[target.index]
        return features, target
    return pd.DataFrame(), pd.Series(dtype=float)

def main():
    features, target = load_features_and_target()
    if features.empty or target.empty:
        print("Features or target is empty.")
        return
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / 'shap_summary_plot.png')
    print(f"SHAP summary plot saved to {RESULTS_ROOT / 'shap_summary_plot.png'}")
    # Force plot (first sample)
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.savefig(RESULTS_ROOT / 'shap_force_plot.png')
    print(f"SHAP force plot saved to {RESULTS_ROOT / 'shap_force_plot.png'}")

if __name__ == "__main__":
    main()
