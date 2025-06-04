from src.data_loader import StudentLifeLoader
from src.feature_engineering import FeatureEngineer
import pandas as pd

def main() -> None:
    """主流程：數據加載、特徵工程、分析。"""
    loader = StudentLifeLoader()
    engineer = FeatureEngineer()

    # 範例：載入GPA資料
    gpa_df: pd.DataFrame = loader.load_gpa_data()
    print("GPA資料：", gpa_df.head())

    # TODO: 載入其他資料、特徵工程、建模分析

if __name__ == "__main__":
    main() 