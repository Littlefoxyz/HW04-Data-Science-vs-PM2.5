from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "pm25_processed.csv"
RESULTS_DIR = BASE_DIR / "output" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1) Load + build daily table
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Bangkok")
df["day"] = df["date"].dt.date

daily = df.groupby("day")[[
    "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "precipitation"
]].mean().reset_index()

daily["day"] = pd.to_datetime(daily["day"])
daily = daily.sort_values("day").reset_index(drop=True)

# ---------------------------------------------------------
# 2) Target: TOMORROW's PM2.5 (shift target back by -1)
# ---------------------------------------------------------
daily["pm2_5_tomorrow"] = daily["pm2_5"].shift(-1)

# ---------------------------------------------------------
# 3) Features: only things known by end of TODAY
# ---------------------------------------------------------
daily["pm2_5_lag1"] = daily["pm2_5"]                      # today's pm2.5
daily["pm2_5_lag2"] = daily["pm2_5"].shift(1)              # yesterday's
daily["pm2_5_roll3"] = daily["pm2_5"].rolling(3).mean()    # 3-day avg
daily["month"] = daily["day"].dt.month

feature_cols = [
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_roll3",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "precipitation", "month",
]

model_df = daily.dropna(subset=feature_cols + ["pm2_5_tomorrow"]).reset_index(drop=True)

X = model_df[feature_cols]
y = model_df["pm2_5_tomorrow"]

# ---------------------------------------------------------
# 4) Time-based split (NOT random) -- last 20% = test
# ---------------------------------------------------------
split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} rows ({model_df['day'].iloc[0].date()} to {model_df['day'].iloc[split_idx-1].date()})")
print(f"Test:  {len(X_test)} rows ({model_df['day'].iloc[split_idx].date()} to {model_df['day'].iloc[-1].date()})")

# ---------------------------------------------------------
# 5) Baseline: persistence (tomorrow = today)
# ---------------------------------------------------------
baseline_pred = X_test["pm2_5_lag1"]  # today's value used as "tomorrow" guess
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

# ---------------------------------------------------------
# 6) Model: Linear Regression + Random Forest
# ---------------------------------------------------------
models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {
    "baseline_persistence": {
        "mae": round(baseline_mae, 3),
        "rmse": round(baseline_rmse, 3),
        "r2": round(baseline_r2, 3),
    }
}

tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {
        "cv_mae_mean": round(-cv_scores.mean(), 3),
        "cv_mae_std": round(cv_scores.std(), 3),
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_r2": round(r2, 3),
        "beats_baseline": bool(mae < baseline_mae),
    }
    print(f"\n{name}: test MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f} "
          f"(baseline MAE={baseline_mae:.3f})")

# ---------------------------------------------------------
# 7) Save results
# ---------------------------------------------------------
with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved metrics -> {RESULTS_DIR / 'metrics.json'}")



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------
# CLASSIFICATION: Will tomorrow exceed 37.5?
# ---------------------------------------------------------
daily["exceeds_tomorrow"] = (daily["pm2_5_tomorrow"] > 37.5).astype(int)

model_df_c = daily.dropna(subset=feature_cols + ["exceeds_tomorrow"]).reset_index(drop=True)
Xc = model_df_c[feature_cols]
yc = model_df_c["exceeds_tomorrow"]

print(f"\nClass balance: {yc.value_counts(normalize=True).round(3).to_dict()}")

split_idx_c = int(len(model_df_c) * 0.8)
Xc_train, Xc_test = Xc.iloc[:split_idx_c], Xc.iloc[split_idx_c:]
yc_train, yc_test = yc.iloc[:split_idx_c], yc.iloc[split_idx_c:]

# --- Baseline: majority class ---
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(Xc_train, yc_train)
baseline_pred_c = dummy.predict(Xc_test)

baseline_c_results = {
    "accuracy": round(accuracy_score(yc_test, baseline_pred_c), 3),
    "precision_exceeds": round(precision_score(yc_test, baseline_pred_c, zero_division=0), 3),
    "recall_exceeds": round(recall_score(yc_test, baseline_pred_c, zero_division=0), 3),
    "f1_exceeds": round(f1_score(yc_test, baseline_pred_c, zero_division=0), 3),
}

# --- Models ---
clf_models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest_clf": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
}

clf_results = {"baseline_majority_class": baseline_c_results}

for name, clf in clf_models.items():
    cv_scores = cross_val_score(clf, Xc_train, yc_train, cv=tscv, scoring="recall")

    clf.fit(Xc_train, yc_train)
    preds = clf.predict(Xc_test)

    cm = confusion_matrix(yc_test, preds).tolist()

    clf_results[name] = {
        "cv_recall_mean": round(cv_scores.mean(), 3),
        "accuracy": round(accuracy_score(yc_test, preds), 3),
        "precision_exceeds": round(precision_score(yc_test, preds, zero_division=0), 3),
        "recall_exceeds": round(recall_score(yc_test, preds, zero_division=0), 3),
        "f1_exceeds": round(f1_score(yc_test, preds, zero_division=0), 3),
        "confusion_matrix": cm,  
        "beats_baseline_recall": bool(recall_score(yc_test, preds, zero_division=0) > baseline_c_results["recall_exceeds"]),
    }
    print(f"\n{name}: recall={clf_results[name]['recall_exceeds']}, "
          f"precision={clf_results[name]['precision_exceeds']}, "
          f"f1={clf_results[name]['f1_exceeds']}")

results["classification"] = clf_results

with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)