from pathlib import Path
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
params_aq = {
    "latitude": 18.7904,
    "longitude": 98.9847,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "dust"],
    "timezone": "Asia/Bangkok",
    "start_date": "2023-01-01",
    "end_date": "2025-04-30",
}
responses_aq = openmeteo.weather_api(url_aq, params=params_aq)
response_aq = responses_aq[0]

hourly_aq = response_aq.Hourly()
data_aq = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly_aq.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_aq.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_aq.Interval()),
        inclusive="left"
    ).tz_convert(response_aq.Timezone().decode()),
    "pm10": hourly_aq.Variables(0).ValuesAsNumpy(),
    "pm2_5": hourly_aq.Variables(1).ValuesAsNumpy(),
    "carbon_monoxide": hourly_aq.Variables(2).ValuesAsNumpy(),
    "dust": hourly_aq.Variables(3).ValuesAsNumpy(),
}

df_aq = pd.DataFrame(data=data_aq)
aq_path = RAW_DIR / "air_quality_raw.csv"
df_aq.to_csv(aq_path, index=False)
print(df_aq.info())







url_wx = "https://archive-api.open-meteo.com/v1/archive"
params_wx = {
    "latitude": 18.7904,
    "longitude": 98.9847,
    "start_date": "2023-01-01",
    "end_date": "2025-04-30",
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"],
    "timezone": "Asia/Bangkok",
}
responses_wx = openmeteo.weather_api(url_wx, params=params_wx)
response_wx = responses_wx[0]

hourly_wx = response_wx.Hourly()
data_wx = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly_wx.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_wx.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_wx.Interval()),
        inclusive="left"
    ).tz_convert(response_wx.Timezone().decode()),
    "temperature_2m": hourly_wx.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly_wx.Variables(1).ValuesAsNumpy(),
    "wind_speed_10m": hourly_wx.Variables(2).ValuesAsNumpy(),
    "precipitation": hourly_wx.Variables(3).ValuesAsNumpy(),
}

df_wx = pd.DataFrame(data=data_wx)
wx_path = RAW_DIR / "weather_raw.csv"
df_wx.to_csv(wx_path, index=False)
print(df_wx.info())


#C6
import requests
import urllib3
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1) ดึง Air4Thai ก่อน
resp = requests.get(
    "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php",
    timeout=30, verify=False
)
data = resp.json()

cm_stations = [
    s for s in data["stations"]
    if "เชียงใหม่" in s.get("areaTH", "") or "เชียงใหม่" in s.get("nameTH", "")
]

# 2) อ่านวัน-เวลาจาก Air4Thai โดยตรง (ไม่ hardcode)
sample_time = cm_stations[0]["AQILast"]["time"]   # เช่น "21:00"
sample_date = cm_stations[0]["AQILast"]["date"]   # เช่น "2026-09-05"

print(f"Air4Thai รายงานที่: {sample_date} {sample_time}")
for s in cm_stations:
    aqi = s["AQILast"]
    print(f"- {s['nameTH']}: PM2.5 = {aqi['PM25']['value']} ({aqi['date']} {aqi['time']})")

# 3) ดึง Open-Meteo ของวันเดียวกัน แล้วกรองด้วยเวลาที่ได้จริงจาก Air4Thai
resp_today = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude": 18.7904, "longitude": 98.9847,
        "start_date": sample_date, "end_date": sample_date,
        "hourly": "pm2_5", "timezone": "Asia/Bangkok",
    },
)
today_data = resp_today.json()
df_today = pd.DataFrame(today_data["hourly"])

match = df_today[df_today["time"].str.contains(sample_time)]
print(f"\nOpen-Meteo ที่ {sample_time}:")
print(match)
