from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "pm25_processed.csv"
FIGURES_DIR = BASE_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE_DIR / "output" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Bangkok")
df['day'] = df['date'].dt.date

daily = df.groupby('day')[['pm2_5', 'pm10','carbon_monoxide','dust','temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation']].mean().reset_index()

# print(daily.info())

# figures1 - When does the season start and end, and does that change from year to year?
plt.figure(figsize=(15, 6))
plt.plot(daily['day'],daily['pm2_5'],label='PM2.5',color = "#D39B52")
plt.plot(daily['day'],daily['pm10'],label='PM10',color = "#227E85")
plt.axhline(37.5,color='red',linestyle='dashdot', linewidth=1, label='Thai 24-hr standard (37.5 µg/m³)')
plt.title('Daily PM2.5 and PM10 in Chiang Mai (2023-2025)')
plt.xlabel('Day')
plt.ylabel('µg/m³')
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'Daily_PM2.5_and_PM10_in_Chiang_Mai_(2023-2025).png', dpi=300)
plt.close()


# figures2 - Which weather conditions accompany the worst days?
cols = ['pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']
corr = daily[cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap: PM2.5 & Weather Factors')
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'PM2.5_and_Weather_Factors.png', dpi=300)
plt.close()

# figures3 - Weather Conditions on Bad Days vs Normal Days
daily['exceeds_standard'] = daily['pm2_5'] > 37.5
daily['Status'] = daily['exceeds_standard'].map({False: 'Normal Day', True: 'High PM2.5'})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=daily, x='Status', y='wind_speed_10m', ax=axes[0], palette=["#8db5e0", "#ca5471"])
axes[0].set_title('Wind Speed (m/s)')
axes[0].set_xlabel('')
axes[0].set_ylabel('m/s')

sns.boxplot(data=daily, x='Status', y='relative_humidity_2m', ax=axes[1], palette=['#8db5e0', '#ca5471'])
axes[1].set_title('Relative Humidity (%)')
axes[1].set_xlabel('')
axes[1].set_ylabel('%')

sns.boxplot(data=daily, x='Status', y='temperature_2m', ax=axes[2], palette=['#8db5e0', '#ca5471'])
axes[2].set_title('Temperature (°C)')
axes[2].set_xlabel('')
axes[2].set_ylabel('°C')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Weather_Conditions_Boxplot.png', dpi=300)
plt.close()

ttest_results = []

print("\n=== T-Test Results ===")
for col in ['wind_speed_10m', 'relative_humidity_2m', 'temperature_2m']:
    normal = daily[~daily['exceeds_standard']][col].dropna()
    high = daily[daily['exceeds_standard']][col].dropna()
    t, p = stats.ttest_ind(normal, high)
    
    print(f"{col}: t={t:.2f}, p-value={p:.4e}")
    
    ttest_results.append({
        'variable': col,
        't_statistic': round(t, 4),
        'p_value': p,
        'significant_p005': p < 0.05
    })

results_df = pd.DataFrame(ttest_results)
results_df.to_csv(RESULTS_DIR / 'ttest_results.csv', index=False)

# figures4 - How many days per year/month exceed the 37.5 standard, and is the trend going up or down?
daily['dt'] = pd.to_datetime(daily['day'])
daily['year'] = daily['dt'].dt.year
daily['month'] = daily['dt'].dt.month

counts = daily.groupby(['year', 'month'])['exceeds_standard'].sum().reset_index()
counts.rename(columns={'exceeds_standard': 'exceeded_days'}, inplace=True)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=counts, 
    x='month', 
    y='exceeded_days', 
    hue='year', 
    palette='Set2'
)

plt.title('Number of Days Exceeding PM2.5 Standard (> 37.5 µg/m³) by Month', fontsize=12, fontweight='bold')
plt.xlabel('Month (1-12)')
plt.ylabel('Number of Days')
plt.legend(title='Year')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'Exceeded_Days_By_Month.png', dpi=300)
plt.close()