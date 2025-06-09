"""
Streamlit EDA Dashboard for StudentLife Dataset
Run: python -m src.eda.eda_dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
import sys
import os

# 動態載入專案根目錄下的 src 模組
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import StudentLifeLoader
from feature_engineering import FeatureEngineer
from config import RESULTS_ROOT, FIGURES_ROOT

st.set_page_config(page_title="StudentLife EDA Dashboard", layout="wide")
st.title("StudentLife Dataset EDA Dashboard")

# 載入資料
@st.cache_data
def load_data():
    loader = StudentLifeLoader()
    engineer = FeatureEngineer()
    features = engineer.create_all_features(loader.merge_all_data())
    return features

data = load_data()

if data is None or data.empty:
    st.error("無法載入資料，請確認資料集是否齊全。")
    st.stop()

st.subheader("資料總覽")
st.write(data.head())
st.write(f"資料維度: {data.shape}")

# 缺失值視覺化
st.subheader("缺失值分析")
fig, ax = plt.subplots(figsize=(min(16, 0.5+0.5*len(data.columns)), 6))
msno.bar(data, ax=ax)
st.pyplot(fig)

# 欄位選擇
st.subheader("欄位分布與統計")
col = st.selectbox("選擇欄位", data.columns)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data[col].dropna(), kde=True, ax=ax)
ax.set_title(f"Distribution of {col}")
st.pyplot(fig)

# 相關性熱圖
st.subheader("特徵相關性熱圖")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 異常值偵測
st.subheader("異常值偵測 (IQR)")
outlier_counts = {}
for col in data.select_dtypes(include=[np.number]).columns:
    x = data[col]
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier = ((x < lower) | (x > upper)) & (~x.isna())
    outlier_counts[col] = outlier.sum()
outlier_series = pd.Series(outlier_counts)
fig, ax = plt.subplots(figsize=(min(16, 0.5+0.5*len(outlier_series)), 6))
outlier_series[outlier_series>0].sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_title('Outlier Count per Feature')
st.pyplot(fig)

st.success("EDA Dashboard 載入完成！")
