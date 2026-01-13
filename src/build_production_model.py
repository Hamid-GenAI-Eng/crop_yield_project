import pandas as pd
import numpy as np
import joblib
import os

# Scikit-Learn Ecosystem
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration
FAO_FILE = 'data/raw/Production_Crops_Livestock_E_All_Data_NOFLAG.csv'
WEATHER_FILE = 'data/raw/POWER_NASA_DATA.csv'
MODEL_PATH = 'models/crop_yield_model.pkl'

def process_data():
    """
    ETL Pipeline: Merges FAO Crop Data with NASA Weather Data.
    Strategy: Multi-Crop Pooling to increase sample size.
    """
    print("Step 1: Ingesting Raw Data...")
    
    # 1. Load FAO Data (Handle encoding issues automatically)
    try:
        fao_df = pd.read_csv(FAO_FILE, encoding='latin-1')
    except:
        fao_df = pd.read_csv(FAO_FILE, encoding='utf-8')
        
    # 2. Load Weather Data
    weather_df = pd.read_csv(WEATHER_FILE, skiprows=13)

    # 3. Filter for Target Region & Crops
    # We use exact FAO names found in the dataset
    target_crops = ['Wheat', 'Rice', 'Maize (corn)', 'Sugar cane', 'Seed cotton, unginned']
    
    fao_subset = fao_df[
        (fao_df['Area'] == 'Pakistan') & 
        (fao_df['Item'].isin(target_crops))
    ].copy()

    # 4. Reshape FAO Data (Wide to Long)
    print("Step 2: Reshaping Crop Data...")
    year_cols = [c for c in fao_subset.columns if c.startswith('Y')]
    fao_long = fao_subset.melt(
        id_vars=['Area', 'Item', 'Element'], 
        value_vars=year_cols, 
        var_name='Year_Str', 
        value_name='Value'
    )
    fao_long['Year'] = fao_long['Year_Str'].str.replace('Y', '').astype(int)
    
    # Pivot to get columns: Yield | Production | Area
    fao_pivot = fao_long.pivot_table(
        index=['Year', 'Item'], 
        columns='Element', 
        values='Value'
    ).reset_index()

    # 5. Process Weather Data (Quarterly Aggregation)
    print("Step 3: Aggregating Weather Features...")
    weather_long = weather_df.melt(
        id_vars=['PARAMETER', 'YEAR'],
        value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        var_name='Month',
        value_name='Value'
    )
    
    # Map Months to Agricultural Quarters
    # Q1 (Jan-Mar): Rabi Growth / Q3 (Jul-Sep): Kharif Growth
    month_map = {
        'JAN': 'Q1', 'FEB': 'Q1', 'MAR': 'Q1',
        'APR': 'Q2', 'MAY': 'Q2', 'JUN': 'Q2',
        'JUL': 'Q3', 'AUG': 'Q3', 'SEP': 'Q3',
        'OCT': 'Q4', 'NOV': 'Q4', 'DEC': 'Q4'
    }
    weather_long['Quarter'] = weather_long['Month'].map(month_map)
    
    # Aggregate: Sum for Rain, Mean for Temp/Solar
    agg_rules = {'PRECTOTCORR': 'sum', 'T2M_MAX': 'mean', 'T2M_MIN': 'mean', 'RH2M': 'mean'}
    
    weather_feats = []
    for param, agg_func in agg_rules.items():
        subset = weather_long[weather_long['PARAMETER'] == param]
        grouped = subset.groupby(['YEAR', 'Quarter'])['Value'].agg(agg_func).reset_index()
        pivoted = grouped.pivot(index='YEAR', columns='Quarter', values='Value')
        pivoted.columns = [f"{param}_{col}" for col in pivoted.columns]
        weather_feats.append(pivoted)
        
    weather_final = pd.concat(weather_feats, axis=1).reset_index()
    weather_final.rename(columns={'YEAR': 'Year'}, inplace=True)

    # 6. Final Merge
    merged_df = pd.merge(fao_pivot, weather_final, on='Year', how='inner')
    
    # 7. Lag Feature (The "Memory" of the Soil)
    merged_df = merged_df.sort_values(['Item', 'Year'])
    merged_df['Yield_Lag1'] = merged_df.groupby('Item')['Yield'].shift(1)
    
    # Drop rows with missing values (First year of data has no lag)
    final_df = merged_df.dropna(subset=['Yield', 'Yield_Lag1'])
    
    print(f"Step 4: Data Engineering Complete. Final Shape: {final_df.shape}")
    return final_df

def train_production_model():
    # 1. Get Data
    df = process_data()
    
    # 2. Define Features
    # Weather Columns + Historical Yield + CROP TYPE
    weather_cols = [c for c in df.columns if 'Q1' in c or 'Q2' in c or 'Q3' in c or 'Q4' in c]
    
    features = weather_cols + ['Yield_Lag1', 'Item']
    target = 'Yield'
    
    X = df[features]
    y = df[target]
    
    # 3. Build Professional Pipeline
    # We use ColumnTransformer to handle Numeric vs Categorical data separately
    numeric_features = weather_cols + ['Yield_Lag1']
    categorical_features = ['Item']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Choice: Random Forest
    # Why? It is robust to the scale differences between Sugarcane (High Yield) and Cotton (Low Yield).
    # Stacking is unnecessary complexity here and risks overfitting on small data.
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))
    ])
    
    # 4. Validation (Cross-Validation)
    print("Step 5: Validating Model Performance...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate R2 (Accuracy)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    print(f" > Average CV R2 Score: {np.mean(r2_scores):.4f} (Target: > 0.85)")
    print(f" > Average MAE Error: {-np.mean(mae_scores):.2f} hg/ha")
    
    # 5. Final Training
    print("Step 6: Training Final Production Model...")
    model.fit(X, y)
    
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, MODEL_PATH)
    print(f"Success! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_production_model()