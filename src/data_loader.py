"""
Data loading module for the StudentLife dataset.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from src.config import DATA_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentLifeLoader:
    """Class for loading and preprocessing StudentLife dataset."""
    
    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to the StudentLife dataset. If None, uses default from config.
        """
        self.data_root = data_root or DATA_ROOT
        self._validate_data_root()
        
    def _validate_data_root(self) -> None:
        """Validate that the data root exists and contains expected files."""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
            
        required_files = [
            "user_info.csv",
            "education/grades.csv",
            "survey/PHQ-9.csv",
            "survey/PerceivedStressScale.csv",
            "EMA/response/Mood"
        ]
        
        for file in required_files:
            if not (self.data_root / file).exists():
                logger.warning(f"Expected file not found: {file}")
    
    def load_user_info(self) -> pd.DataFrame:
        """
        Load basic user information.
        
        Returns:
            DataFrame containing user demographics and basic info
        """
        path = self.data_root / "user_info.csv"
        if not path.exists():
            logger.warning(f"user_info.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def load_gpa_data(self) -> pd.DataFrame:
        """
        Load GPA and academic performance data.
        
        Returns:
            DataFrame containing academic performance metrics
        """
        path = self.data_root / "education" / "grades.csv"
        if not path.exists():
            logger.warning(f"grades.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def load_phq9_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 depression assessment data.
        
        Returns:
            DataFrame containing PHQ-9 scores
        """
        path = self.data_root / "survey" / "PHQ-9.csv"
        if not path.exists():
            logger.warning(f"PHQ-9.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def load_pss_data(self) -> pd.DataFrame:
        """
        Load Perceived Stress Scale (PSS) data.
        
        Returns:
            DataFrame containing PSS scores
        """
        path = self.data_root / "survey" / "PerceivedStressScale.csv"
        if not path.exists():
            logger.warning(f"PerceivedStressScale.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def load_ema_data(self, subfolder: str = "Mood") -> pd.DataFrame:
        """
        批次讀取 EMA/response/ 下指定子資料夾的所有 csv，合併為一個 DataFrame。
        """
        ema_dir = self.data_root / "EMA" / "response" / subfolder
        if not ema_dir.exists():
            logger.warning(f"EMA subfolder not found: {ema_dir}")
            return pd.DataFrame()
        all_csv = list(ema_dir.glob("*.csv"))
        if not all_csv:
            logger.warning(f"No csv files in {ema_dir}")
            return pd.DataFrame()
        df_list = [pd.read_csv(f) for f in all_csv]
        return pd.concat(df_list, ignore_index=True)
    
    def load_sensor_data(self, sensor_type: str) -> pd.DataFrame:
        """
        批次讀取感測器資料夾（如 gps、phonelock、activity、audio、conversation）下所有 csv，合併為一個 DataFrame。
        """
        sensor_dir = self.data_root / "sensing" / sensor_type
        if not sensor_dir.exists():
            logger.warning(f"Sensor data directory not found: {sensor_dir}")
            return pd.DataFrame()
        all_csv = list(sensor_dir.glob("*.csv"))
        if not all_csv:
            logger.warning(f"No csv files in {sensor_dir}")
            return pd.DataFrame()
        df_list = []
        for f in tqdm(all_csv, desc=f"Loading {sensor_type}"):
            try:
                df = pd.read_csv(f)
                # 自動加上 user_id 欄位（從檔名推斷）
                user_id = f.stem.split('_')[-1]
                df['user_id'] = user_id
                df_list.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
        if not df_list:
            logger.warning(f"No valid csv files loaded in {sensor_dir}")
            return pd.DataFrame()
        return pd.concat(df_list, ignore_index=True)
    
    def load_activity_data(self) -> pd.DataFrame:
        """
        批次讀取 sensing/activity/ 下所有學生的 activity 檔案，合併為一個 DataFrame。
        """
        return self.load_sensor_data('activity')
    
    def load_audio_data(self) -> pd.DataFrame:
        """
        批次讀取 sensing/audio/ 下所有學生的 audio 檔案，合併為一個 DataFrame。
        """
        return self.load_sensor_data('audio')

    def load_conversation_data(self) -> pd.DataFrame:
        """
        批次讀取 sensing/conversation/ 下所有學生的 conversation 檔案，合併為一個 DataFrame。
        """
        return self.load_sensor_data('conversation')

    def load_ema_mood_data(self) -> pd.DataFrame:
        """
        批次讀取 EMA/response/Mood/ 下所有 csv，合併為一個 DataFrame。
        """
        return self.load_ema_data('Mood')

    def load_ema_sleep_data(self) -> pd.DataFrame:
        """
        批次讀取 EMA/response/Sleep/ 下所有 csv，合併為一個 DataFrame。
        """
        return self.load_ema_data('Sleep')
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Merge all available data sources into a single DataFrame.
        
        Returns:
            Merged DataFrame containing all available data
        """
        # Load all data sources
        user_info = self.load_user_info()
        gpa_data = self.load_gpa_data()
        phq9_data = self.load_phq9_data()
        pss_data = self.load_pss_data()
        ema_data = self.load_ema_data()
        
        # Merge data sources
        # Implementation will depend on the specific merge logic needed
        # This is a placeholder for the actual implementation
        return pd.DataFrame()

    def load_bluetooth_data(self) -> pd.DataFrame:
        return self.load_sensor_data('bluetooth')

    def load_wifi_location_data(self) -> pd.DataFrame:
        return self.load_sensor_data('wifi_location')

    def load_wifi_data(self) -> pd.DataFrame:
        return self.load_sensor_data('wifi')

    def load_phonecharge_data(self) -> pd.DataFrame:
        return self.load_sensor_data('phonecharge')

    def load_dark_data(self) -> pd.DataFrame:
        return self.load_sensor_data('dark')

    def load_call_log_data(self) -> pd.DataFrame:
        return self.load_sensor_data('call_log')

    def load_sms_data(self) -> pd.DataFrame:
        return self.load_sensor_data('sms')

    def load_app_usage_data(self) -> pd.DataFrame:
        return self.load_sensor_data('app_usage')

    def load_calendar_data(self) -> pd.DataFrame:
        return self.load_sensor_data('calendar')

    def load_dinning_data(self) -> pd.DataFrame:
        return self.load_sensor_data('dinning')

    def load_panas_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "panas.csv"
        if not path.exists():
            logger.warning(f"panas.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_psqi_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "psqi.csv"
        if not path.exists():
            logger.warning(f"psqi.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_bigfive_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "BigFive.csv"
        if not path.exists():
            logger.warning(f"BigFive.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_flourishing_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "FlourishingScale.csv"
        if not path.exists():
            logger.warning(f"FlourishingScale.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_loneliness_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "LonelinessScale.csv"
        if not path.exists():
            logger.warning(f"LonelinessScale.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_vr12_data(self) -> pd.DataFrame:
        path = self.data_root / "survey" / "vr_12.csv"
        if not path.exists():
            logger.warning(f"vr_12.csv not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    # 自動產生所有 EMA 子資料夾 loader
    def load_ema_activity_data(self) -> pd.DataFrame:
        return self.load_ema_data('Activity')
    def load_ema_stress_data(self) -> pd.DataFrame:
        return self.load_ema_data('Stress')
    def load_ema_social_data(self) -> pd.DataFrame:
        return self.load_ema_data('Social')
    def load_ema_pam_data(self) -> pd.DataFrame:
        return self.load_ema_data('PAM')
    def load_ema_exercise_data(self) -> pd.DataFrame:
        return self.load_ema_data('Exercise')
    def load_ema_events_data(self) -> pd.DataFrame:
        return self.load_ema_data('Events')
    def load_ema_study_spaces_data(self) -> pd.DataFrame:
        return self.load_ema_data('Study Spaces')
    def load_ema_administration_response_data(self) -> pd.DataFrame:
        return self.load_ema_data("Administration's response")
    def load_ema_green_key1_data(self) -> pd.DataFrame:
        return self.load_ema_data('Green Key 1')
    def load_ema_green_key2_data(self) -> pd.DataFrame:
        return self.load_ema_data('Green Key 2')
    def load_ema_cancelled_classes_data(self) -> pd.DataFrame:
        return self.load_ema_data('Cancelled Classes')
    def load_ema_class2_data(self) -> pd.DataFrame:
        return self.load_ema_data('Class 2')
    def load_ema_dartmouth_now_data(self) -> pd.DataFrame:
        return self.load_ema_data('Dartmouth now')
    def load_ema_dimensions_protestors_data(self) -> pd.DataFrame:
        return self.load_ema_data('Dimensions protestors')
    def load_ema_dining_halls_data(self) -> pd.DataFrame:
        return self.load_ema_data('Dining Halls')
    def load_ema_lab_data(self) -> pd.DataFrame:
        return self.load_ema_data('Lab')
    def load_ema_mood1_data(self) -> pd.DataFrame:
        return self.load_ema_data('Mood 1')
    def load_ema_mood2_data(self) -> pd.DataFrame:
        return self.load_ema_data('Mood 2')
    def load_ema_behavior_data(self) -> pd.DataFrame:
        return self.load_ema_data('Behavior')
    def load_ema_boston_bombing_data(self) -> pd.DataFrame:
        return self.load_ema_data('Boston Bombing')
    def load_ema_dimensions_data(self) -> pd.DataFrame:
        return self.load_ema_data('Dimensions')
    def load_ema_do_campbells_jokes_suck_data(self) -> pd.DataFrame:
        return self.load_ema_data("Do Campbell's jokes suck_")
    def load_ema_comment_data(self) -> pd.DataFrame:
        return self.load_ema_data('Comment')
    def load_ema_sleep_data(self) -> pd.DataFrame:
        return self.load_ema_data('Sleep')
    def load_ema_social2_data(self) -> pd.DataFrame:
        return self.load_ema_data('Social')
    def load_ema_stress2_data(self) -> pd.DataFrame:
        return self.load_ema_data('Stress')