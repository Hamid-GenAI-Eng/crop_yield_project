from data_basic_info import fao_df, weather_df
import pandas as pd
# 1. FAO Data Processing
# Filter for Pakistan
fao_pak = fao_df[fao_df['Area'] == 'Pakistan'].copy()

# Melt Year columns
year_cols = [c for c in fao_pak.columns if c.startswith('Y')]
id_vars = ['Area', 'Item', 'Element']
fao_long = fao_pak.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year_Str', value_name='Value')

# Clean Year
fao_long['Year'] = fao_long['Year_Str'].str.replace('Y', '').astype(int)

# Pivot Elements (Yield, Production, Area harvested)
fao_pivot = fao_long.pivot_table(
    index=['Year', 'Item'],
    columns='Element',
    values='Value'
).reset_index()

# Rename columns for clarity (remove flattened index if any)
fao_pivot.columns.name = None

# Filter for major crops to check data quality (Optional, but good for inspection)
# Common crops: Wheat, Rice, paddy, Maize, Cotton lint, Sugarcane
major_crops = ['Wheat', 'Rice, paddy', 'Maize', 'Sugarcane', 'Cotton seed', 'Cotton lint']
fao_final = fao_pivot[fao_pivot['Item'].isin(major_crops)].copy()


# 2. Weather Data Processing
# The weather file is: PARAMETER, YEAR, JAN, FEB ... ANN
# We need to reshape this.

# Define Seasons for Pakistan context
# Winter (Rabi): Dec, Jan, Feb, Mar (Growth for Wheat)
# Summer (Kharif): Jun, Jul, Aug, Sep (Growth for Rice/Cotton)
# We will create simplified seasonal aggregates:
# Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec

weather_long = weather_df.melt(
    id_vars=['PARAMETER', 'YEAR'],
    value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
    var_name='Month',
    value_name='Value'
)

# Map Month to Season/Quarter
month_map = {
    'JAN': 'Q1', 'FEB': 'Q1', 'MAR': 'Q1',
    'APR': 'Q2', 'MAY': 'Q2', 'JUN': 'Q2',
    'JUL': 'Q3', 'AUG': 'Q3', 'SEP': 'Q3',
    'OCT': 'Q4', 'NOV': 'Q4', 'DEC': 'Q4'
}
weather_long['Quarter'] = weather_long['Month'].map(month_map)

# Aggregation logic:
# Rain (PRECTOTCORR) -> Sum
# Others (Temp, Humidity, Solar) -> Mean

agg_rules = {
    'PRECTOTCORR': 'sum',
    'T2M_MAX': 'mean',
    'T2M_MIN': 'mean',
    'RH2M': 'mean',
    'ALLSKY_SFC_SW_DWN': 'mean'
}

# Apply aggregation
weather_feats = []

for param, agg_func in agg_rules.items():
    # Filter for specific parameter
    subset = weather_long[weather_long['PARAMETER'] == param]
    
    # Group by Year and Quarter
    grouped = subset.groupby(['YEAR', 'Quarter'])['Value'].agg(agg_func).reset_index()
    
    # Pivot to get Year | Q1_Param | Q2_Param ...
    pivoted = grouped.pivot(index='YEAR', columns='Quarter', values='Value')
    pivoted.columns = [f"{param}_{col}" for col in pivoted.columns]
    
    weather_feats.append(pivoted)

# Also add Annual Averages from the original file (column 'ANN')
annual_feats = weather_df.pivot(index='YEAR', columns='PARAMETER', values='ANN')
annual_feats.columns = [f"{col}_ANN" for col in annual_feats.columns]
weather_feats.append(annual_feats)

# Merge all weather features
weather_final = pd.concat(weather_feats, axis=1).reset_index()
weather_final.rename(columns={'YEAR': 'Year'}, inplace=True)


# 3. Final Merge
# Left join FAO with Weather (preserve Crop data structure)
merged_df = pd.merge(fao_final, weather_final, on='Year', how='inner')

print("Merged Data Head:")
print(merged_df.head())
print("\nMerged Data Info:")
print(merged_df.info())

# Save to CSV
merged_df.to_csv('merged_crop_weather_data.csv', index=False)
print("\nSaved 'merged_crop_weather_data.csv'")