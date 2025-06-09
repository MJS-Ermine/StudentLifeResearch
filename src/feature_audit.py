import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from src.config import DATA_ROOT, RESULTS_ROOT

REPORT_PATH = RESULTS_ROOT / 'feature_audit_report.md'
FIG_PATH = RESULTS_ROOT / 'figures/feature_missing_heatmap.png'


def scan_all_csv_columns(data_root: Path) -> dict:
    """
    掃描 data_root 下所有 csv 檔案，回傳 {檔案路徑: 欄位名稱list}
    """
    col_info = {}
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.csv'):
                path = Path(root) / f
                try:
                    df = pd.read_csv(path, nrows=100)
                    col_info[str(path.relative_to(data_root))] = list(df.columns)
                except Exception as e:
                    col_info[str(path.relative_to(data_root))] = [f"[讀取失敗: {e}]"]
    return col_info

def scan_column_coverage(data_root: Path) -> pd.DataFrame:
    """
    掃描所有 csv，統計每個欄位的非空比例與型態。
    回傳 DataFrame: index=檔案, columns=欄位, value=非空比例
    """
    records = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.csv'):
                path = Path(root) / f
                try:
                    df = pd.read_csv(path)
                    for col in df.columns:
                        nonnull = df[col].notna().sum()
                        total = len(df)
                        dtype = str(df[col].dtype)
                        records.append({
                            'file': str(path.relative_to(data_root)),
                            'column': col.strip(),
                            'nonnull': nonnull,
                            'total': total,
                            'coverage': nonnull/total if total else 0,
                            'dtype': dtype
                        })
                except Exception as e:
                    records.append({'file': str(path.relative_to(data_root)), 'column': '[讀取失敗]', 'nonnull': 0, 'total': 0, 'coverage': 0, 'dtype': str(e)})
    return pd.DataFrame(records)

def generate_audit_report():
    col_info = scan_all_csv_columns(DATA_ROOT)
    coverage_df = scan_column_coverage(DATA_ROOT)
    # 欄位清單 markdown
    md = ['# StudentLife 欄位覆蓋率與欄位清單報告\n']
    md.append('## 所有檔案欄位清單')
    for file, cols in col_info.items():
        md.append(f'- **{file}**: {", ".join(cols)}')
    # 欄位覆蓋率統計
    md.append('\n## 欄位覆蓋率統計 (前20)')
    top_cov = coverage_df.groupby('column')['coverage'].mean().sort_values(ascending=False).head(20)
    for col, cov in top_cov.items():
        md.append(f'- {col}: 平均覆蓋率 {cov:.2%}')
    # 缺失熱圖
    pivot = coverage_df.pivot(index='file', columns='column', values='coverage').fillna(0)
    plt.figure(figsize=(min(20, 0.5+0.5*len(pivot.columns)), min(12, 0.5+0.5*len(pivot))))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': '非空覆蓋率'})
    plt.title('StudentLife 欄位缺失熱圖')
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    plt.close()
    md.append(f'\n![](figures/feature_missing_heatmap.png)')
    # 輸出 markdown
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"[INFO] 欄位覆蓋率報告已產生: {REPORT_PATH}")
    print(f"[INFO] 欄位缺失熱圖已產生: {FIG_PATH}")

if __name__ == "__main__":
    generate_audit_report() 