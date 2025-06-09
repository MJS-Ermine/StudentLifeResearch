"""
Feature engineering module for the StudentLife dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from src.config import FEATURE_PARAMS
from src.data_loader import StudentLifeLoader
from scipy.stats import entropy as shannon_entropy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_column_mapping(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """
    自動將 df 欄位名稱依 mapping_dict 進行對應與轉換，支援去除空白、大小寫不敏感。
    Args:
        df: 原始 DataFrame
        mapping_dict: {標準欄位: [可能的別名列表]}
    Returns:
        欄位已標準化的 DataFrame
    """
    col_map = {col.strip().lower(): col for col in df.columns}
    for std_col, aliases in mapping_dict.items():
        found = None
        for alias in aliases:
            alias_key = alias.strip().lower()
            if alias_key in col_map:
                found = col_map[alias_key]
                break
        if found and std_col not in df.columns:
            df = df.rename(columns={found: std_col})
    return df

class FeatureEngineer:
    """Class for engineering features from StudentLife dataset."""
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            params: Dictionary of feature engineering parameters. If None, uses default from config.
        """
        self.params = params or FEATURE_PARAMS
        
    def extract_sleep_features(self, sleep_df: pd.DataFrame) -> pd.DataFrame:
        if sleep_df.empty:
            logger.warning("Sleep data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'sleep_duration': ['sleep_duration', 'duration'],
            'time_in_bed': ['time_in_bed'],
            'is_weekend': ['is_weekend']
        }
        sleep_df = robust_column_mapping(sleep_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in sleep_df.columns]
        if missing_cols:
            logger.warning(f"Sleep data missing columns: {missing_cols}. Columns: {sleep_df.columns.tolist()}")
            return pd.DataFrame()
        features = sleep_df.groupby('user_id').agg(
            avg_sleep_duration = ('sleep_duration', 'mean'),
            sleep_regularity = ('sleep_duration', lambda x: x.std()/x.mean() if x.mean() else np.nan),
            sleep_efficiency = ('sleep_duration', lambda x: x.sum()/sleep_df.loc[x.index, 'time_in_bed'].sum() if sleep_df.loc[x.index, 'time_in_bed'].sum() else np.nan),
            weekend_sleep_diff = ('sleep_duration', lambda x: sleep_df.loc[x.index & sleep_df['is_weekend'], 'sleep_duration'].mean() - sleep_df.loc[x.index & ~sleep_df['is_weekend'], 'sleep_duration'].mean())
        ).reset_index()
        return features
    
    def extract_phone_usage_features(self, phonelock_df: pd.DataFrame) -> pd.DataFrame:
        if phonelock_df.empty:
            logger.warning("Phonelock data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'event_type': ['event_type'],
            'timestamp': ['timestamp', 'time', 'datetime', 'start', 'end']
        }
        phonelock_df = robust_column_mapping(phonelock_df, mapping)
        # 自動偵測時間欄位
        time_col = None
        for col in ['timestamp', 'time', 'datetime', 'start', 'end']:
            if col in phonelock_df.columns:
                time_col = col
                break
        if time_col is None:
            logger.warning(f"Phonelock data has no time column (tried: timestamp, time, datetime, start, end). Columns: {phonelock_df.columns.tolist()}")
            return pd.DataFrame()
        phonelock_df['datetime'] = pd.to_datetime(phonelock_df[time_col], errors='coerce')
        phonelock_df['date'] = phonelock_df['datetime'].dt.date
        phonelock_df['hour'] = phonelock_df['datetime'].dt.hour
        # event_type 欄位容錯
        event_col = 'event_type' if 'event_type' in phonelock_df.columns else None
        if event_col is None:
            logger.warning(f"Phonelock data has no event_type column. Columns: {phonelock_df.columns.tolist()}")
            return pd.DataFrame()
        unlocks = phonelock_df[phonelock_df[event_col] == 'unlock']
        features = unlocks.groupby('user_id').agg(
            daily_unlocks = ('date', 'nunique'),
            night_unlocks = ('hour', lambda x: ((x >= 22) | (x < 6)).sum())
        ).reset_index()
        return features
    
    def extract_activity_features(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        if activity_df.empty:
            logger.warning("Activity data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'steps': ['steps', 'step_count'],
            'gps_radius': ['gps_radius'],
            'is_outside': ['is_outside'],
            'activity_inference': ['activity_inference', ' activity inference']
        }
        activity_df = robust_column_mapping(activity_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in activity_df.columns]
        if missing_cols:
            logger.warning(f"Activity data missing columns: {missing_cols}. Columns: {activity_df.columns.tolist()}")
            return pd.DataFrame()
        features = activity_df.groupby('user_id').agg(
            avg_daily_steps = ('steps', 'mean'),
            activity_range = ('gps_radius', 'mean'),
            outing_frequency = ('is_outside', 'mean')
        ).reset_index()
        return features
    
    def extract_mental_health_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract mental health-related features.
        
        Args:
            data: DataFrame containing mental health data
            
        Returns:
            DataFrame containing mental health features
        """
        features = pd.DataFrame()
        
        # Calculate PHQ-9 scores
        features['phq9_score'] = data.groupby('user_id')['phq9_total'].mean()
        
        # Calculate stress levels
        features['stress_level'] = data.groupby('user_id')['pss_score'].mean()
        
        # Calculate mood variability
        features['mood_variability'] = data.groupby('user_id')['ema_mood'].std()
        
        # Calculate mood-behavior consistency
        features['mood_behavior_consistency'] = data.groupby('user_id').apply(
            lambda x: x['ema_mood'].corr(x['activity_level'])
        )
        
        return features
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from the input data.
        
        Args:
            data: DataFrame containing all raw data
            
        Returns:
            DataFrame containing all engineered features
        """
        # Extract features from different domains
        sleep_features = self.extract_sleep_features(data)
        phone_features = self.extract_phone_usage_features(data)
        activity_features = self.extract_activity_features(data)
        mental_health_features = self.extract_mental_health_features(data)
        
        # Merge all features
        all_features = pd.concat([
            sleep_features,
            phone_features,
            activity_features,
            mental_health_features
        ], axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(all_features.mean())
        
        return all_features

    def extract_audio_features(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        if audio_df.empty:
            logger.warning("Audio data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'conversation_duration': ['conversation_duration', 'duration'],
            'noise_level': ['noise_level']
        }
        audio_df = robust_column_mapping(audio_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in audio_df.columns]
        if missing_cols:
            logger.warning(f"Audio data missing columns: {missing_cols}. Columns: {audio_df.columns.tolist()}")
            return pd.DataFrame()
        features = audio_df.groupby('user_id').agg(
            avg_conversation_time = ('conversation_duration', 'mean'),
            avg_noise_level = ('noise_level', 'mean')
        ).reset_index()
        return features

    def extract_conversation_features(self, conv_df: pd.DataFrame) -> pd.DataFrame:
        if conv_df.empty:
            logger.warning("Conversation data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'duration': ['duration', 'conversation_duration']
        }
        conv_df = robust_column_mapping(conv_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in conv_df.columns]
        if missing_cols:
            logger.warning(f"Conversation data missing columns: {missing_cols}. Columns: {conv_df.columns.tolist()}")
            return pd.DataFrame()
        features = conv_df.groupby('user_id').agg(
            total_conversation_time = ('duration', 'sum'),
            avg_conversation_time = ('duration', 'mean')
        ).reset_index()
        return features

    def extract_ema_mood_features(self, ema_mood_df: pd.DataFrame) -> pd.DataFrame:
        if ema_mood_df.empty:
            logger.warning("EMA Mood data is empty.")
            return pd.DataFrame()
        required_cols = ['user_id', 'mood_score']
        missing_cols = [col for col in required_cols if col not in ema_mood_df.columns]
        if missing_cols:
            logger.warning(f"EMA Mood data missing columns: {missing_cols}. Columns: {ema_mood_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_mood_df.groupby('user_id').agg(
            avg_mood = ('mood_score', 'mean'),
            mood_variability = ('mood_score', 'std')
        ).reset_index()
        return features

    def extract_ema_sleep_features(self, ema_sleep_df: pd.DataFrame) -> pd.DataFrame:
        if ema_sleep_df.empty:
            logger.warning("EMA Sleep data is empty.")
            return pd.DataFrame()
        required_cols = ['user_id', 'sleep_quality']
        missing_cols = [col for col in required_cols if col not in ema_sleep_df.columns]
        if missing_cols:
            logger.warning(f"EMA Sleep data missing columns: {missing_cols}. Columns: {ema_sleep_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_sleep_df.groupby('user_id').agg(
            avg_sleep_quality = ('sleep_quality', 'mean')
        ).reset_index()
        return features

    def extract_bluetooth_features(self, bluetooth_df: pd.DataFrame) -> pd.DataFrame:
        if bluetooth_df.empty:
            logger.warning("Bluetooth data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'device_count': ['device_count']
        }
        bluetooth_df = robust_column_mapping(bluetooth_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in bluetooth_df.columns]
        if missing_cols:
            logger.warning(f"Bluetooth data missing columns: {missing_cols}. Columns: {bluetooth_df.columns.tolist()}")
            return pd.DataFrame()
        features = bluetooth_df.groupby('user_id').agg(
            avg_daily_contacts = ('device_count', 'mean')
        ).reset_index()
        return features

    def extract_wifi_features(self, wifi_df: pd.DataFrame) -> pd.DataFrame:
        if wifi_df.empty:
            logger.warning("WiFi data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'location_id': ['location_id']
        }
        wifi_df = robust_column_mapping(wifi_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in wifi_df.columns]
        if missing_cols:
            logger.warning(f"WiFi data missing columns: {missing_cols}. Columns: {wifi_df.columns.tolist()}")
            return pd.DataFrame()
        features = wifi_df.groupby('user_id').agg(
            unique_locations = ('location_id', 'nunique')
        ).reset_index()
        return features

    def extract_phonecharge_features(self, phonecharge_df: pd.DataFrame) -> pd.DataFrame:
        if phonecharge_df.empty:
            logger.warning("Phonecharge data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'timestamp': ['timestamp', 'start', 'end', 'datetime']
        }
        phonecharge_df = robust_column_mapping(phonecharge_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in phonecharge_df.columns]
        if missing_cols:
            logger.warning(f"Phonecharge data missing columns: {missing_cols}. Columns: {phonecharge_df.columns.tolist()}")
            return pd.DataFrame()
        phonecharge_df['datetime'] = pd.to_datetime(phonecharge_df['timestamp'], errors='coerce')
        phonecharge_df['hour'] = phonecharge_df['datetime'].dt.hour
        features = phonecharge_df.groupby('user_id').agg(
            night_charge_freq = ('hour', lambda x: ((x >= 22) | (x < 6)).sum())
        ).reset_index()
        return features

    def extract_dark_features(self, dark_df: pd.DataFrame) -> pd.DataFrame:
        if dark_df.empty:
            logger.warning("Dark data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'dark_duration': ['dark_duration', 'duration']
        }
        dark_df = robust_column_mapping(dark_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in dark_df.columns]
        if missing_cols:
            logger.warning(f"Dark data missing columns: {missing_cols}. Columns: {dark_df.columns.tolist()}")
            return pd.DataFrame()
        features = dark_df.groupby('user_id').agg(
            avg_night_dark_duration = ('dark_duration', 'mean')
        ).reset_index()
        return features

    def extract_app_usage_features(self, app_usage_df: pd.DataFrame) -> pd.DataFrame:
        if app_usage_df.empty:
            logger.warning("App usage data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'app_category': ['app_category', 'category'],
            'usage_time': ['usage_time', 'duration']
        }
        app_usage_df = robust_column_mapping(app_usage_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in app_usage_df.columns]
        if missing_cols:
            logger.warning(f"App usage data missing columns: {missing_cols}. Columns: {app_usage_df.columns.tolist()}")
            return pd.DataFrame()
        features = app_usage_df.groupby(['user_id', 'app_category']).agg(
            total_usage = ('usage_time', 'sum')
        ).unstack(fill_value=0).reset_index()
        return features

    def extract_dinning_features(self, dinning_df: pd.DataFrame) -> pd.DataFrame:
        if dinning_df.empty:
            logger.warning("Dinning data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'meal_time': ['meal_time', 'time', 'datetime', 'start', 'end']
        }
        dinning_df = robust_column_mapping(dinning_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in dinning_df.columns]
        if missing_cols:
            logger.warning(f"Dinning data missing columns: {missing_cols}. Columns: {dinning_df.columns.tolist()}")
            return pd.DataFrame()
        dinning_df['datetime'] = pd.to_datetime(dinning_df['meal_time'], errors='coerce')
        dinning_df['date'] = dinning_df['datetime'].dt.date
        features = dinning_df.groupby('user_id').agg(
            avg_meals_per_day = ('date', 'nunique')
        ).reset_index()
        return features

    def extract_panas_features(self, panas_df: pd.DataFrame) -> pd.DataFrame:
        if panas_df.empty:
            logger.warning("PANAS data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'positive': ['positive'],
            'negative': ['negative']
        }
        panas_df = robust_column_mapping(panas_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in panas_df.columns]
        if missing_cols:
            logger.warning(f"PANAS data missing columns: {missing_cols}. Columns: {panas_df.columns.tolist()}")
            return pd.DataFrame()
        features = panas_df.groupby('user_id').agg(
            avg_positive = ('positive', 'mean'),
            avg_negative = ('negative', 'mean')
        ).reset_index()
        return features

    def extract_psqi_features(self, psqi_df: pd.DataFrame) -> pd.DataFrame:
        if psqi_df.empty:
            logger.warning("PSQI data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'psqi_score': ['psqi_score']
        }
        psqi_df = robust_column_mapping(psqi_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in psqi_df.columns]
        if missing_cols:
            logger.warning(f"PSQI data missing columns: {missing_cols}. Columns: {psqi_df.columns.tolist()}")
            return pd.DataFrame()
        features = psqi_df.groupby('user_id').agg(
            avg_psqi = ('psqi_score', 'mean')
        ).reset_index()
        return features

    def extract_bigfive_features(self, bigfive_df: pd.DataFrame) -> pd.DataFrame:
        if bigfive_df.empty:
            logger.warning("BigFive data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'openness': ['openness'],
            'conscientiousness': ['conscientiousness'],
            'extraversion': ['extraversion'],
            'agreeableness': ['agreeableness'],
            'neuroticism': ['neuroticism']
        }
        bigfive_df = robust_column_mapping(bigfive_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in bigfive_df.columns]
        if missing_cols:
            logger.warning(f"BigFive data missing columns: {missing_cols}. Columns: {bigfive_df.columns.tolist()}")
            return pd.DataFrame()
        features = bigfive_df.groupby('user_id').agg(
            openness = ('openness', 'mean'),
            conscientiousness = ('conscientiousness', 'mean'),
            extraversion = ('extraversion', 'mean'),
            agreeableness = ('agreeableness', 'mean'),
            neuroticism = ('neuroticism', 'mean')
        ).reset_index()
        return features

    def extract_flourishing_features(self, flourishing_df: pd.DataFrame) -> pd.DataFrame:
        if flourishing_df.empty:
            logger.warning("Flourishing data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'flourishing_score': ['flourishing_score']
        }
        flourishing_df = robust_column_mapping(flourishing_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in flourishing_df.columns]
        if missing_cols:
            logger.warning(f"Flourishing data missing columns: {missing_cols}. Columns: {flourishing_df.columns.tolist()}")
            return pd.DataFrame()
        features = flourishing_df.groupby('user_id').agg(
            avg_flourishing = ('flourishing_score', 'mean')
        ).reset_index()
        return features

    def extract_loneliness_features(self, loneliness_df: pd.DataFrame) -> pd.DataFrame:
        if loneliness_df.empty:
            logger.warning("Loneliness data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'loneliness_score': ['loneliness_score']
        }
        loneliness_df = robust_column_mapping(loneliness_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in loneliness_df.columns]
        if missing_cols:
            logger.warning(f"Loneliness data missing columns: {missing_cols}. Columns: {loneliness_df.columns.tolist()}")
            return pd.DataFrame()
        features = loneliness_df.groupby('user_id').agg(
            avg_loneliness = ('loneliness_score', 'mean')
        ).reset_index()
        return features

    def extract_vr12_features(self, vr12_df: pd.DataFrame) -> pd.DataFrame:
        if vr12_df.empty:
            logger.warning("VR-12 data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'vr12_score': ['vr12_score']
        }
        vr12_df = robust_column_mapping(vr12_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in vr12_df.columns]
        if missing_cols:
            logger.warning(f"VR-12 data missing columns: {missing_cols}. Columns: {vr12_df.columns.tolist()}")
            return pd.DataFrame()
        features = vr12_df.groupby('user_id').agg(
            avg_vr12 = ('vr12_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_activity_features(self, ema_activity_df: pd.DataFrame) -> pd.DataFrame:
        if ema_activity_df.empty:
            logger.warning("EMA Activity data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'activity_score': ['activity_score']
        }
        ema_activity_df = robust_column_mapping(ema_activity_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in ema_activity_df.columns]
        if missing_cols:
            logger.warning(f"EMA Activity data missing columns: {missing_cols}. Columns: {ema_activity_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_activity_df.groupby('user_id').agg(
            avg_activity = ('activity_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_stress_features(self, ema_stress_df: pd.DataFrame) -> pd.DataFrame:
        if ema_stress_df.empty:
            logger.warning("EMA Stress data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'stress_score': ['stress_score']
        }
        ema_stress_df = robust_column_mapping(ema_stress_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in ema_stress_df.columns]
        if missing_cols:
            logger.warning(f"EMA Stress data missing columns: {missing_cols}. Columns: {ema_stress_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_stress_df.groupby('user_id').agg(
            avg_stress = ('stress_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_social_features(self, ema_social_df: pd.DataFrame) -> pd.DataFrame:
        if ema_social_df.empty:
            logger.warning("EMA Social data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'social_score': ['social_score']
        }
        ema_social_df = robust_column_mapping(ema_social_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in ema_social_df.columns]
        if missing_cols:
            logger.warning(f"EMA Social data missing columns: {missing_cols}. Columns: {ema_social_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_social_df.groupby('user_id').agg(
            avg_social = ('social_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_pam_features(self, ema_pam_df: pd.DataFrame) -> pd.DataFrame:
        if ema_pam_df.empty:
            logger.warning("EMA PAM data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid'],
            'pam_score': ['pam_score']
        }
        ema_pam_df = robust_column_mapping(ema_pam_df, mapping)
        required_cols = list(mapping.keys())
        missing_cols = [col for col in required_cols if col not in ema_pam_df.columns]
        if missing_cols:
            logger.warning(f"EMA PAM data missing columns: {missing_cols}. Columns: {ema_pam_df.columns.tolist()}")
            return pd.DataFrame()
        features = ema_pam_df.groupby('user_id').agg(
            avg_pam = ('pam_score', 'mean')
        ).reset_index()
        return features

    def extract_activity_inference_features(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        """
        針對 activity_inference 欄位自動推導多種統計特徵。
        包含：各活動類型比例、活動切換次數、最常見活動、多樣性指標（Shannon entropy、unique count）、one-hot encoding。
        """
        if activity_df.empty:
            logger.warning("Activity data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid', 'id'],
            'activity_inference': ['activity_inference', ' activity inference']
        }
        activity_df = robust_column_mapping(activity_df, mapping)
        if 'user_id' not in activity_df.columns or 'activity_inference' not in activity_df.columns:
            logger.warning(f"Activity data missing user_id/activity_inference. Columns: {activity_df.columns.tolist()}")
            return pd.DataFrame()
        activity_df['activity_inference'] = activity_df['activity_inference'].astype(str).str.strip()
        activity_counts = activity_df.groupby(['user_id', 'activity_inference']).size().unstack(fill_value=0)
        activity_props = activity_counts.div(activity_counts.sum(axis=1), axis=0)
        activity_props.columns = [f'activity_prop_{c}' for c in activity_props.columns]
        # 多樣性指標
        activity_entropy = activity_props.apply(lambda x: shannon_entropy(x + 1e-9), axis=1).rename('activity_entropy')
        activity_unique = activity_counts.astype(bool).sum(axis=1).rename('activity_unique_count')
        # 活動切換次數
        def count_switches(x):
            return (x != x.shift()).sum() - 1 if len(x) > 1 else 0
        switches = activity_df.groupby('user_id')['activity_inference'].apply(count_switches).rename('activity_switch_count')
        # 最常見活動
        most_common = activity_df.groupby('user_id')['activity_inference'].agg(lambda x: x.value_counts().idxmax()).rename('most_common_activity')
        # one-hot encoding
        onehot = pd.get_dummies(activity_df.set_index('user_id')['activity_inference']).groupby('user_id').sum()
        onehot.columns = [f'activity_onehot_{c}' for c in onehot.columns]
        features = pd.concat([activity_props, activity_entropy, activity_unique, switches, most_common, onehot], axis=1).reset_index()
        return features

    def extract_audio_inference_features(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        """
        針對 audio_inference 欄位自動推導多種統計特徵。
        包含：各音訊狀態比例、對話段落數、最常見音訊狀態、多樣性指標（Shannon entropy、unique count）、one-hot encoding。
        OOM防呆：僅保留必要欄位，one-hot僅針對前10名。
        """
        if audio_df.empty:
            logger.warning("Audio data is empty.")
            return pd.DataFrame()
        mapping = {
            'user_id': ['user_id', 'uid', 'id'],
            'audio_inference': ['audio_inference', ' audio inference']
        }
        audio_df = robust_column_mapping(audio_df, mapping)
        if 'user_id' not in audio_df.columns or 'audio_inference' not in audio_df.columns:
            logger.warning(f"Audio data missing user_id/audio_inference. Columns: {audio_df.columns.tolist()}")
            return pd.DataFrame()
        # 僅保留必要欄位
        audio_df = audio_df[['user_id', 'audio_inference']].copy()
        # 先補齊缺失再轉字串
        audio_df['audio_inference'] = audio_df['audio_inference'].fillna('unknown').astype(str).str.strip().str.lower()
        # 只保留出現次數前10名，其餘歸為 'other'
        top10 = audio_df['audio_inference'].value_counts().nlargest(10).index
        audio_df.loc[~audio_df['audio_inference'].isin(top10), 'audio_inference'] = 'other'
        audio_counts = audio_df.groupby(['user_id', 'audio_inference']).size().unstack(fill_value=0)
        audio_props = audio_counts.div(audio_counts.sum(axis=1), axis=0)
        audio_props.columns = [f'audio_prop_{c}' for c in audio_props.columns]
        # 多樣性指標
        audio_entropy = audio_props.apply(lambda x: shannon_entropy(x + 1e-9), axis=1).rename('audio_entropy')
        audio_unique = audio_counts.astype(bool).sum(axis=1).rename('audio_unique_count')
        # 對話段落數
        conv_count = audio_df[audio_df['audio_inference'].isin(['conversation', 'speech'])].groupby('user_id').size().rename('conversation_segment_count')
        # 最常見音訊狀態
        most_common = audio_df.groupby('user_id')['audio_inference'].agg(lambda x: x.value_counts().idxmax()).rename('most_common_audio_state')
        # one-hot encoding（僅前10名+other）
        onehot = pd.get_dummies(audio_df.set_index('user_id')['audio_inference']).groupby('user_id').sum()
        onehot.columns = [f'audio_onehot_{c}' for c in onehot.columns]
        features = pd.concat([audio_props, audio_entropy, audio_unique, conv_count, most_common, onehot], axis=1).reset_index()
        return features

    def extract_ema_summary_features(self, ema_df: pd.DataFrame, prefix: str = "ema") -> pd.DataFrame:
        """
        自動產生 EMA 問卷 summary 特徵：均值、標準差、極端值比例、多樣性指標（Shannon entropy、unique count）。
        prefix: 特徵前綴（如 ema_activity, ema_stress...）
        """
        if ema_df.empty or 'user_id' not in ema_df.columns:
            logger.warning(f"EMA data is empty or missing user_id.")
            return pd.DataFrame()
        # 只處理數值型欄位
        num_cols = [c for c in ema_df.columns if c not in ['user_id', 'timestamp', 'resp_time'] and np.issubdtype(ema_df[c].dtype, np.number)]
        if not num_cols:
            return pd.DataFrame()
        agg = ema_df.groupby('user_id')[num_cols].agg(['mean', 'std', 'min', 'max'])
        agg.columns = [f'{prefix}_{col}_{stat}' for col, stat in agg.columns]
        # 極端值比例（如 min/max）
        extreme = ema_df.groupby('user_id')[num_cols].apply(lambda x: ((x == x.min()) | (x == x.max())).mean()).add_prefix(f'{prefix}_extreme_')
        # 多樣性指標
        entropy_df = ema_df.groupby('user_id')[num_cols].apply(lambda x: shannon_entropy(x.value_counts(normalize=True) + 1e-9)).add_prefix(f'{prefix}_entropy_')
        unique_df = ema_df.groupby('user_id')[num_cols].nunique().add_prefix(f'{prefix}_unique_')
        features = pd.concat([agg, extreme, entropy_df, unique_df], axis=1).reset_index()
        return features

def merge_all_features(loader: 'StudentLifeLoader', engineer: 'FeatureEngineer') -> pd.DataFrame:
    """
    自動合併所有特徵（所有感測器、問卷、EMA），以 user_id 為 key。
    自動對所有 EMA 問卷產生 summary/多樣性特徵。
    """
    # 讀取所有資料
    sleep_df = loader.load_sensor_data('sleep')
    phonelock_df = loader.load_sensor_data('phonelock')
    activity_df = loader.load_activity_data()
    audio_df = loader.load_audio_data()
    conv_df = loader.load_conversation_data()
    bluetooth_df = loader.load_bluetooth_data()
    wifi_df = loader.load_wifi_data()
    phonecharge_df = loader.load_phonecharge_data()
    dark_df = loader.load_dark_data()
    app_usage_df = loader.load_app_usage_data()
    dinning_df = loader.load_dinning_data()
    panas_df = loader.load_panas_data()
    psqi_df = loader.load_psqi_data()
    bigfive_df = loader.load_bigfive_data()
    flourishing_df = loader.load_flourishing_data()
    loneliness_df = loader.load_loneliness_data()
    vr12_df = loader.load_vr12_data()
    ema_activity_df = loader.load_ema_activity_data()
    ema_stress_df = loader.load_ema_stress_data()
    ema_social_df = loader.load_ema_social_data()
    ema_pam_df = loader.load_ema_pam_data()
    # 自動產生所有 EMA summary/多樣性特徵
    ema_summary_list = []
    for name, df in zip([
        'ema_activity', 'ema_stress', 'ema_social', 'ema_pam'],
        [ema_activity_df, ema_stress_df, ema_social_df, ema_pam_df]):
        ema_summary = engineer.extract_ema_summary_features(df, prefix=name)
        if ema_summary is not None and not ema_summary.empty:
            ema_summary_list.append(ema_summary)
    # 計算所有特徵
    features_list = [
        engineer.extract_sleep_features(sleep_df),
        engineer.extract_phone_usage_features(phonelock_df),
        engineer.extract_activity_features(activity_df),
        engineer.extract_audio_features(audio_df),
        engineer.extract_conversation_features(conv_df),
        engineer.extract_bluetooth_features(bluetooth_df),
        engineer.extract_wifi_features(wifi_df),
        engineer.extract_phonecharge_features(phonecharge_df),
        engineer.extract_dark_features(dark_df),
        engineer.extract_app_usage_features(app_usage_df),
        engineer.extract_dinning_features(dinning_df),
        engineer.extract_panas_features(panas_df),
        engineer.extract_psqi_features(psqi_df),
        engineer.extract_bigfive_features(bigfive_df),
        engineer.extract_flourishing_features(flourishing_df),
        engineer.extract_loneliness_features(loneliness_df),
        engineer.extract_vr12_features(vr12_df),
        engineer.extract_ema_activity_features(ema_activity_df),
        engineer.extract_ema_stress_features(ema_stress_df),
        engineer.extract_ema_social_features(ema_social_df),
        engineer.extract_ema_pam_features(ema_pam_df),
        engineer.extract_activity_inference_features(activity_df),
        engineer.extract_audio_inference_features(audio_df)
    ] + ema_summary_list
    features = None
    for feat in features_list:
        if feat is not None and not feat.empty:
            if features is None:
                features = feat
            else:
                features = pd.merge(features, feat, on='user_id', how='outer')
    return features if features is not None else pd.DataFrame()