"""
Main script for running the StudentLife analysis pipeline.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import StudentLifeLoader
from .feature_engineering import FeatureEngineer
from .config import MODEL_PARAMS, RESULTS_ROOT, FIGURES_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the StudentLife dataset."""
    logger.info("Loading data...")
    loader = StudentLifeLoader()
    data = loader.merge_all_data()
    
    if data.empty:
        raise ValueError("No data loaded. Please check the data directory.")
    
    logger.info(f"Loaded data shape: {data.shape}")
    return data

def create_features(data: pd.DataFrame):
    """Create features from the raw data."""
    logger.info("Creating features...")
    engineer = FeatureEngineer()
    features = engineer.create_all_features(data)
    
    logger.info(f"Created features shape: {features.shape}")
    return features

def train_models(features: pd.DataFrame, target: pd.Series):
    """Train and evaluate models."""
    logger.info("Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'random_forest': RandomForestRegressor(**MODEL_PARAMS['random_forest']),
        'ridge': Ridge(**MODEL_PARAMS['ridge'])
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Save feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_
            })
            importance.to_csv(RESULTS_ROOT / f'{name}_feature_importance.csv')
    
    return results

def create_visualizations(features: pd.DataFrame, target: pd.Series):
    """Create and save visualizations."""
    logger.info("Creating visualizations...")
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation = pd.concat([features, target], axis=1).corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'correlation_heatmap.png')
    plt.close()
    
    # Feature distributions
    for feature in features.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=features, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / f'{feature}_distribution.png')
        plt.close()

def main():
    """Main function to run the analysis pipeline."""
    try:
        # Setup
        setup_directories()
        
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Create features
        features = create_features(data)
        
        # Get target variable
        target = data['gpa']
        
        # Train models
        results = train_models(features, target)
        
        # Create visualizations
        create_visualizations(features, target)
        
        # Save results
        pd.DataFrame(results).to_csv(RESULTS_ROOT / 'model_results.csv')
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()