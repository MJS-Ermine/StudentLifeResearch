import pandas as pd
from pathlib import Path
from config import DATA_ROOT

class StudentLifeLoader:
    """StudentLife 數據集加載器。"""
    def __init__(self) -> None:
        self.data_root: Path = DATA_ROOT

    def load_user_info(self) -> pd.DataFrame:
        """載入用戶基本信息。"""
        return pd.read_csv(self.data_root / "user_info.csv")

    def load_gpa_data(self) -> pd.DataFrame:
        """載入GPA數據。"""
        return pd.read_csv(self.data_root / "education" / "grades.csv")

    # 可依需求擴充更多載入方法 