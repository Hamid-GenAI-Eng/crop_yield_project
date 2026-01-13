import pandas as pd
import io

try:
    fao_df = pd.read_csv('../data/raw/Production_Crops_Livestock_E_All_Data_NOFLAG.csv', encoding='latin-1')
except:
    fao_df = pd.read_csv('../data/raw/Production_Crops_Livestock_E_All_Data_NOFLAG.csv', encoding='utf-8')


weather_df = pd.read_csv('POWER_NASA_DATA.csv', skiprows=13)

print("FAO Data Head:")
print(fao_df.head())
print("\nFAO Data Info:")
print(fao_df.info())

print("\nWeather Data Head:")
print(weather_df.head())
print("\nWeather Data Info:")
print(weather_df.info())