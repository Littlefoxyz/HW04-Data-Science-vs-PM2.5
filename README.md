# PM2.5 in Northern Thailand: From Raw Data to a Recommendation

Every year between roughly January and April, the air in Northern Thailand becomes dangerous to breathe. Agricultural residue burning, forest fires and transboundary smoke combine with a stable atmosphere and mountain topography that traps pollution in the valleys. Schools close. Hospital admissions for respiratory illness rise. Tourism falls. Chiang Mai has on several occasions been reported as having the worst urban air quality in the world.

Thailand's 24-hour ambient standard for PM2.5 is 37.5 micrograms per cubic metre. That single number decides whether a warning is issued, whether outdoor activities are cancelled, and whether a day is counted as a violation in the official statistics.

## Report
`report/Report_HOMEWORK 4.pdf`

## Presentation clip
[ใส่ลิงก์ YouTube ตรงนี้]

## How to run (in order)

```bash
pip install -r requirements.txt

python scr/fetch_data.py
python scr/prepare_data.py
python scr/analyse.py
python scr/model.py
```

## Data sources

1. **Open-Meteo Air Quality API** — hourly PM2.5, PM10, carbon_monoxide, dust, 2023-01-01 to 2025-04-30, timezone Asia/Bangkok
2. **Open-Meteo Archive API** — hourly weather (temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation), timezone Asia/Bangkok
3. **Air4Thai** — ground-station snapshot, used as a ground-truth sanity check (see report Section C6)

## Study location

Chiang Mai, Thailand (18.7883, 98.9853)

## Models

- **Regression**: predicts tomorrow's mean PM2.5 (Linear Regression, Random Forest)
- **Classification**: predicts whether tomorrow exceeds the Thai 24-hr standard of 37.5 µg/m³ (Logistic Regression, Random Forest Classifier)

Both use time-based train/test split (not random) and TimeSeriesSplit cross-validation, since the data is temporally ordered.
