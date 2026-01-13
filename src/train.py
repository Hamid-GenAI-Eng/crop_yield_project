import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom engineering from the other file
from preprocessing import AgriculturalFeatureEngineer, get_training_pipeline

# Configuration
DATA_PATH = 'data/processed/merged_crop_weather_data.csv'
MODEL_PATH = 'models/crop_yield_stacking_model.pkl'

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # --- CRITICAL FIX START ---
    # Filter for 'Wheat' only. 
    # Analysis showed 'Cotton seed' has 100% missing Yield values and causes data corruption when shifted.
    df = df[df['Item'] == 'Wheat'].copy()
    # --- CRITICAL FIX END ---

    # SENIOR CHECK: Sort by year
    df = df.sort_values('Year')
    
    features = [
        'PRECTOTCORR_Q1', 'PRECTOTCORR_Q2', 'PRECTOTCORR_Q3', 'PRECTOTCORR_Q4',
        'T2M_MAX_Q1', 'T2M_MAX_Q2', 'T2M_MAX_Q3', 'T2M_MAX_Q4',
        'T2M_MIN_Q1', 'T2M_MIN_Q2', 'T2M_MIN_Q3', 'T2M_MIN_Q4',
        'RH2M_Q1', 'RH2M_Q2', 'RH2M_Q3', 'RH2M_Q4',
        'ALLSKY_SFC_SW_DWN_Q1'
    ]
    target = 'Yield'
    
    # Handle Lag (Memory)
    df['Yield_Lag1'] = df['Yield'].shift(1)
    features.append('Yield_Lag1')
    
    # Robust Drop: Ensure we drop rows where EITHER Lag OR Target is missing
    df = df.dropna(subset=['Yield_Lag1', target])
    
    X = df[features]
    y = df[target]
    
    print(f"Data loaded: {len(df)} rows ready for training.")
    return X, y

def train_model():
    # 1. Load Data
    X, y = load_data(DATA_PATH)
    
    # 2. Define Base Models
    # XGBoost: The Kaggle winner. Good at capturing complex patterns.
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # Random Forest: Robust, hard to overfit.
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # Ridge: Linear baseline. Keeps the other two "grounded".
    ridge = Ridge(alpha=1.0)
    
    estimators = [
        ('xgb', xgb),
        ('rf', rf),
        ('ridge', ridge)
    ]
    
    # 3. Define Stacking Regressor
    # The 'final_estimator' takes the outputs of xgb, rf, and ridge and makes the final decision
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        passthrough=False # Meta-learner only sees predictions, not original features
    )
    
    # 4. Build the Full Pipeline
    # Raw Data -> Feature Engineering -> Imputation/Scaling -> Stacking Model
    final_pipeline = Pipeline([
        ('feature_engineering', AgriculturalFeatureEngineer()),
        ('preprocessing', get_training_pipeline()),
        ('model', stacking_regressor)
    ])
    
    # 5. K-Fold Cross Validation (The "Truth Serum")
    print("Starting 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # We use negative MSE because cross_val_score tries to maximize the score
    scores = cross_val_score(final_pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    
    print(f"CV RMSE Scores: {rmse_scores}")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} (hg/ha)")
    
    # 6. Final Training & Serialization
    print("Training final model on full dataset...")
    final_pipeline.fit(X, y)
    
    # Calculate training R2 for sanity check
    y_pred = final_pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Training R2 Score: {r2:.4f}")
    
    # Save the pipeline
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
    print("Ready for Production.")

if __name__ == "__main__":
    train_model()