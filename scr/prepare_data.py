from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df_aq = pd.read_csv(RAW_DIR / 'air_quality_raw.csv')
df_w = pd.read_csv(RAW_DIR / 'weather_raw.csv')

df_aq["date"] = pd.to_datetime(df_aq["date"], utc=True).dt.tz_convert("Asia/Bangkok")
df_w["date"] = pd.to_datetime(df_w["date"], utc=True).dt.tz_convert("Asia/Bangkok")

print('\n-------- AQ --------\n')

print("First timestamp:", df_aq["date"].min())
print("Last timestamp:", df_aq["date"].max())
print("Timezone info:", df_aq["date"].dt.tz)
print('AQ:', df_aq.shape)
print(df_aq.info())
missing_aq = pd.DataFrame({
    'missing_count': df_aq.isnull().sum(),
    'missing_pct (%)': (df_aq.isnull().sum() / len(df_aq)) * 100
})
print(missing_aq)

print('\n-------- Weather --------\n')

print("First timestamp:", df_w["date"].min())
print("Last timestamp:", df_w["date"].max())
print("Timezone info:", df_w["date"].dt.tz)
print('Weather:', df_w.shape)
print(df_w.info())
missing_w = pd.DataFrame({
    'missing_count': df_w.isnull().sum(),
    'missing_pct (%)': (df_w.isnull().sum() / len(df_w)) * 100
})
print(missing_w)


df_pm5_processed = pd.merge(df_aq, df_w, on='date', how='inner')
print('\n-------- After Join --------\n')

print("First timestamp:", df_pm5_processed["date"].min())
print("Last timestamp:", df_pm5_processed["date"].max())
print("Timezone info:", df_pm5_processed["date"].dt.tz)
print(df_pm5_processed.info())
print(df_pm5_processed.shape)

print('\n-------- C3: Missing Values Inspection --------\n')

missing_summary = pd.DataFrame({
    'missing_count': df_pm5_processed.isnull().sum(),
    'missing_pct (%)': (df_pm5_processed.isnull().sum() / len(df_pm5_processed)) * 100
})
print(missing_summary)

output_path = PROCESSED_DIR / 'pm25_processed.csv'
df_pm5_processed.to_csv(output_path, index=False)

