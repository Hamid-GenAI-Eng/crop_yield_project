import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class AgriculturalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that generates agronomic features.
    This ensures that 'Pro' features are calculated on the fly during prediction.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Avoid SettingWithCopy warnings
        X = X.copy()
        
        # 1. Biological Interaction: Heat Stress Index (Humid Heat)
        # High heat + High humidity = Fungal diseases or Heat stress
        if 'T2M_MAX_Q1' in X.columns and 'RH2M_Q1' in X.columns:
            X['Q1_Heat_Stress'] = X['T2M_MAX_Q1'] * X['RH2M_Q1']

        # 2. Resource Efficiency: Rain per unit of Temperature
        # Did it rain enough to counteract the evaporation from heat?
        if 'PRECTOTCORR_Q1' in X.columns and 'T2M_MAX_Q1' in X.columns:
            # Add small epsilon (0.1) to avoid division by zero
            X['Q1_Rain_Temp_Ratio'] = X['PRECTOTCORR_Q1'] / (X['T2M_MAX_Q1'] + 0.1)
            
        # 3. Solar Energy Intake (Photosynthesis Potential)
        if 'ALLSKY_SFC_SW_DWN_Q1' in X.columns:
             X['Q1_Solar_Accumulation'] = X['ALLSKY_SFC_SW_DWN_Q1'] * 90 # Approx days in Q1

        return X

def get_training_pipeline():
    """
    Returns a unified pipeline: Imputation -> Feature Engineering -> Scaling
    """
    pipeline = Pipeline([
        # Step 1: Handle missing values (Impute with Median)
        ('imputer', SimpleImputer(strategy='median')),
        
        # Step 2: Scale features (Models like Ridge/SVR require this)
        ('scaler', StandardScaler())
    ])
    
    return pipeline