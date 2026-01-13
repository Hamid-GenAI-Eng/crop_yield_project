from data__prep import merged_df
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Focus on Wheat for EDA (It's a major Rabi crop in Pakistan)
wheat_df = merged_df[merged_df['Item'] == 'Wheat'].copy()

# Feature Engineering: Lag Yield (The "Memory" of the soil/farmer)
wheat_df = wheat_df.sort_values('Year')
wheat_df['Yield_Lag1'] = wheat_df['Yield'].shift(1)

# Feature Engineering: Simple Interaction Terms
# Q1 is Jan-Mar (Critical growth/grain filling for Wheat in Pakistan)
wheat_df['Q1_Heat_Index'] = wheat_df['T2M_MAX_Q1'] * wheat_df['RH2M_Q1'] # Humid Heat
wheat_df['Q1_Rain_Temp_Ratio'] = wheat_df['PRECTOTCORR_Q1'] / (wheat_df['T2M_MAX_Q1'] + 0.1)

# Select columns for Correlation Analysis
cols_to_corr = [
    'Yield', 
    'Yield_Lag1',
    'PRECTOTCORR_Q1', 'T2M_MAX_Q1', 'T2M_MIN_Q1', # Winter/Spring Weather
    'PRECTOTCORR_Q4', 'T2M_MAX_Q4',               # Sowing Season Weather (Oct-Dec)
    'Q1_Heat_Index'
]

# Compute Correlation
corr_matrix = wheat_df[cols_to_corr].corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix: Wheat Yield vs Seasonal Weather (Pakistan)')
plt.savefig('wheat_correlation_heatmap.png')

# Plot Yield over Time to see trend
plt.figure(figsize=(10, 5))
plt.plot(wheat_df['Year'], wheat_df['Yield'], marker='o', linestyle='-')
plt.title('Wheat Yield in Pakistan (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Yield (hg/ha)')
plt.grid(True)
plt.savefig('../data/processed/wheat_yield_trend.png')

print("EDA Analysis Complete.")
print(wheat_df[['Year', 'Yield', 'Yield_Lag1', 'Q1_Heat_Index']].head())