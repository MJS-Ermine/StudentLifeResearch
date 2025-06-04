# StudentLife 行為分析專案

## 專案簡介
本專案旨在分析大學生日常行為與學業表現之關聯，基於 StudentLife dataset 進行特徵工程與機器學習建模。

## 安裝方式
```bash
pip install -r requirements.txt
```

## 資料集取得
- 請至 [StudentLife Dataset](https://studentlife.cs.dartmouth.edu/dataset.html) 申請並下載數據集
- 將解壓後的資料放置於 `data/dataset/` 目錄

## 執行方式
```bash
python -m src.main
```

## 專案結構
- `src/`：主要程式碼
- `data/`：原始數據
- `results/`：分析結果
- `notebooks/`：Jupyter 探索

## 其他
- 請依實際數據格式調整 loader 與特徵工程
