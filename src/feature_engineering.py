"""
Feature engineering module for the StudentLife dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from .config import FEATURE_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # 假設 sleep_df 有 user_id, sleep_duration, time_in_bed, is_weekend
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
        # 假設 phonelock_df 有 user_id, timestamp, event_type
        # 需先轉換 timestamp 為日期與時段
        phonelock_df['datetime'] = pd.to_datetime(phonelock_df['timestamp'], errors='coerce')
        phonelock_df['date'] = phonelock_df['datetime'].dt.date
        phonelock_df['hour'] = phonelock_df['datetime'].dt.hour
        unlocks = phonelock_df[phonelock_df['event_type'] == 'unlock']
        features = unlocks.groupby('user_id').agg(
            daily_unlocks = ('date', 'nunique'),
            night_unlocks = ('hour', lambda x: ((x >= 22) | (x < 6)).sum())
        ).reset_index()
        return features
    
    def extract_activity_features(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        if activity_df.empty:
            logger.warning("Activity data is empty.")
            return pd.DataFrame()
        # 假設 activity_df 有 user_id, steps, timestamp, gps_radius, is_outside
        activity_df['datetime'] = pd.to_datetime(activity_df['timestamp'], errors='coerce')
        activity_df['date'] = activity_df['datetime'].dt.date
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
        # 假設 audio_df 有 user_id, conversation_duration, noise_level
        features = audio_df.groupby('user_id').agg(
            avg_conversation_time = ('conversation_duration', 'mean'),
            avg_noise_level = ('noise_level', 'mean')
        ).reset_index()
        return features

    def extract_conversation_features(self, conv_df: pd.DataFrame) -> pd.DataFrame:
        if conv_df.empty:
            logger.warning("Conversation data is empty.")
            return pd.DataFrame()
        # 假設 conv_df 有 user_id, duration
        features = conv_df.groupby('user_id').agg(
            total_conversation_time = ('duration', 'sum'),
            avg_conversation_time = ('duration', 'mean')
        ).reset_index()
        return features

    def extract_ema_mood_features(self, ema_mood_df: pd.DataFrame) -> pd.DataFrame:
        if ema_mood_df.empty:
            logger.warning("EMA Mood data is empty.")
            return pd.DataFrame()
        # 假設 ema_mood_df 有 user_id, mood_score
        features = ema_mood_df.groupby('user_id').agg(
            avg_mood = ('mood_score', 'mean'),
            mood_variability = ('mood_score', 'std')
        ).reset_index()
        return features

    def extract_ema_sleep_features(self, ema_sleep_df: pd.DataFrame) -> pd.DataFrame:
        if ema_sleep_df.empty:
            logger.warning("EMA Sleep data is empty.")
            return pd.DataFrame()
        # 假設 ema_sleep_df 有 user_id, sleep_quality
        features = ema_sleep_df.groupby('user_id').agg(
            avg_sleep_quality = ('sleep_quality', 'mean')
        ).reset_index()
        return features

    def extract_bluetooth_features(self, bluetooth_df: pd.DataFrame) -> pd.DataFrame:
        if bluetooth_df.empty:
            logger.warning("Bluetooth data is empty.")
            return pd.DataFrame()
        # 假設 bluetooth_df 有 user_id, device_count
        features = bluetooth_df.groupby('user_id').agg(
            avg_daily_contacts = ('device_count', 'mean')
        ).reset_index()
        return features

    def extract_wifi_features(self, wifi_df: pd.DataFrame) -> pd.DataFrame:
        if wifi_df.empty:
            logger.warning("WiFi data is empty.")
            return pd.DataFrame()
        # 假設 wifi_df 有 user_id, location_id
        features = wifi_df.groupby('user_id').agg(
            unique_locations = ('location_id', 'nunique')
        ).reset_index()
        return features

    def extract_phonecharge_features(self, phonecharge_df: pd.DataFrame) -> pd.DataFrame:
        if phonecharge_df.empty:
            logger.warning("Phonecharge data is empty.")
            return pd.DataFrame()
        # 假設 phonecharge_df 有 user_id, timestamp, is_charging
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
        # 假設 dark_df 有 user_id, dark_duration
        features = dark_df.groupby('user_id').agg(
            avg_night_dark_duration = ('dark_duration', 'mean')
        ).reset_index()
        return features

    def extract_app_usage_features(self, app_usage_df: pd.DataFrame) -> pd.DataFrame:
        if app_usage_df.empty:
            logger.warning("App usage data is empty.")
            return pd.DataFrame()
        # 假設 app_usage_df 有 user_id, app_category, usage_time
        features = app_usage_df.groupby(['user_id', 'app_category']).agg(
            total_usage = ('usage_time', 'sum')
        ).unstack(fill_value=0).reset_index()
        return features

    def extract_dinning_features(self, dinning_df: pd.DataFrame) -> pd.DataFrame:
        if dinning_df.empty:
            logger.warning("Dinning data is empty.")
            return pd.DataFrame()
        # 假設 dinning_df 有 user_id, meal_time
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
        # 假設 panas_df 有 user_id, positive, negative
        features = panas_df.groupby('user_id').agg(
            avg_positive = ('positive', 'mean'),
            avg_negative = ('negative', 'mean')
        ).reset_index()
        return features

    def extract_psqi_features(self, psqi_df: pd.DataFrame) -> pd.DataFrame:
        if psqi_df.empty:
            logger.warning("PSQI data is empty.")
            return pd.DataFrame()
        # 假設 psqi_df 有 user_id, psqi_score
        features = psqi_df.groupby('user_id').agg(
            avg_psqi = ('psqi_score', 'mean')
        ).reset_index()
        return features

    def extract_bigfive_features(self, bigfive_df: pd.DataFrame) -> pd.DataFrame:
        if bigfive_df.empty:
            logger.warning("BigFive data is empty.")
            return pd.DataFrame()
        # 假設 bigfive_df 有 user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism
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
        # 假設 flourishing_df 有 user_id, flourishing_score
        features = flourishing_df.groupby('user_id').agg(
            avg_flourishing = ('flourishing_score', 'mean')
        ).reset_index()
        return features

    def extract_loneliness_features(self, loneliness_df: pd.DataFrame) -> pd.DataFrame:
        if loneliness_df.empty:
            logger.warning("Loneliness data is empty.")
            return pd.DataFrame()
        # 假設 loneliness_df 有 user_id, loneliness_score
        features = loneliness_df.groupby('user_id').agg(
            avg_loneliness = ('loneliness_score', 'mean')
        ).reset_index()
        return features

    def extract_vr12_features(self, vr12_df: pd.DataFrame) -> pd.DataFrame:
        if vr12_df.empty:
            logger.warning("VR-12 data is empty.")
            return pd.DataFrame()
        # 假設 vr12_df 有 user_id, vr12_score
        features = vr12_df.groupby('user_id').agg(
            avg_vr12 = ('vr12_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_activity_features(self, ema_activity_df: pd.DataFrame) -> pd.DataFrame:
        if ema_activity_df.empty:
            logger.warning("EMA Activity data is empty.")
            return pd.DataFrame()
        # 假設 ema_activity_df 有 user_id, activity_score
        features = ema_activity_df.groupby('user_id').agg(
            avg_activity = ('activity_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_stress_features(self, ema_stress_df: pd.DataFrame) -> pd.DataFrame:
        if ema_stress_df.empty:
            logger.warning("EMA Stress data is empty.")
            return pd.DataFrame()
        # 假設 ema_stress_df 有 user_id, stress_score
        features = ema_stress_df.groupby('user_id').agg(
            avg_stress = ('stress_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_social_features(self, ema_social_df: pd.DataFrame) -> pd.DataFrame:
        if ema_social_df.empty:
            logger.warning("EMA Social data is empty.")
            return pd.DataFrame()
        # 假設 ema_social_df 有 user_id, social_score
        features = ema_social_df.groupby('user_id').agg(
            avg_social = ('social_score', 'mean')
        ).reset_index()
        return features

    def extract_ema_pam_features(self, ema_pam_df: pd.DataFrame) -> pd.DataFrame:
        if ema_pam_df.empty:
            logger.warning("EMA PAM data is empty.")
            return pd.DataFrame()
        # 假設 ema_pam_df 有 user_id, pam_score
        features = ema_pam_df.groupby('user_id').agg(
            avg_pam = ('pam_score', 'mean')
        ).reset_index()
        return features

def merge_all_features(loader: 'StudentLifeLoader', engineer: 'FeatureEngineer') -> pd.DataFrame:
    """
    自動合併所有特徵（所有感測器、問卷、EMA），以 user_id 為 key。
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
        engineer.extract_ema_pam_features(ema_pam_df)
    ]
    features = None
    for feat in features_list:
        if feat is not None and not feat.empty:
            if features is None:
                features = feat
            else:
                features = pd.merge(features, feat, on='user_id', how='outer')
    return features if features is not None else pd.DataFrame()