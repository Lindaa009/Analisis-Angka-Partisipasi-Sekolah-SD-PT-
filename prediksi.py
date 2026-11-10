"""Analisis Angka Partisipasi Sekolah (APS) Indonesia, 2021–2025."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

file_path = BASE_DIR / "aps_indonesia_2021_2025.xlsx"

df = pd.read_excel(file_path)

print("="*50)
print("DATASET")
print("="*50)

print(df.head())

print("\n" + "="*50)
print("INFO DATA")
print("="*50)

df.info()

print("\n" + "="*50)
print("MISSING VALUE")
print("="*50)

print(df.isnull().sum())

print("\nJumlah data duplikat :", df.duplicated().sum())

df = df.drop_duplicates()

kolom_numerik = [
    'Tingkat Kemiskinan Perkotaan (%)',
    'Tingkat Kemiskinan Perdesaan (%)',
    'PDRB per Kapita',
    'Jumlah Sekolah SD',
    'Jumlah Sekolah SMP',
    'Jumlah Sekolah SMA',
    'Jumlah Sekolah KULIAH',
    'Rasio Guru-Murid SD',
    'Rasio Guru-Murid SMP',
    'Rasio Guru-Murid SMA',
    'Rasio Guru-Murid KULIAH',
    'APS SD',
    'APS SMP',
    'APS SMA',
    'APS KULIAH'
]

for col in kolom_numerik:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(',', '.', regex=False)
        .replace('-', pd.NA)
    )

    df[col] = pd.to_numeric(df[col], errors='coerce').astype("float64")

for col in kolom_numerik:
    df[col] = df[col].fillna(df[col].mean())

print("\n" + "="*50)
print("INFO DATA SETELAH CLEANING")
print("="*50)

df.info()

print("\n" + "="*50)
print("CEK NaN")
print("="*50)

print(df.isnull().sum())

print("\n" + "="*50)
print("STATISTIK DESKRIPTIF")
print("="*50)

print(df.describe())

plt.figure(figsize=(10,6))

plt.plot(
    df.groupby('Tahun')['APS SD'].mean(),
    marker='o',
    label='APS SD'
)

plt.plot(
    df.groupby('Tahun')['APS SMP'].mean(),
    marker='o',
    label='APS SMP'
)

plt.plot(
    df.groupby('Tahun')['APS SMA'].mean(),
    marker='o',
    label='APS SMA'
)

plt.plot(
    df.groupby('Tahun')['APS KULIAH'].mean(),
    marker='o',
    label='APS KULIAH'
)

plt.title("Tren APS 2021-2025")
plt.xlabel("Tahun")
plt.ylabel("APS")
plt.legend()
plt.grid(True)

plt.savefig(FIGURE_DIR / "aps_trend.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14,10))

sns.heatmap(
    df[kolom_numerik].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Heatmap Korelasi")

plt.savefig(FIGURE_DIR / "aps_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10,6))

sns.boxplot(data=df[['APS SD','APS SMP','APS SMA','APS KULIAH']])

plt.title("Boxplot APS")

plt.savefig(FIGURE_DIR / "aps_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

X = df[[
    'Tingkat Kemiskinan Perkotaan (%)',
    'Tingkat Kemiskinan Perdesaan (%)',
    'PDRB per Kapita',
    'Jumlah Sekolah SD',
    'Jumlah Sekolah SMP',
    'Jumlah Sekolah SMA',
    'Jumlah Sekolah KULIAH',
    'Rasio Guru-Murid SD',
    'Rasio Guru-Murid SMP',
    'Rasio Guru-Murid SMA',
    'Rasio Guru-Murid KULIAH'
]]

targets = {
    'APS SD': df['APS SD'],
    'APS SMP': df['APS SMP'],
    'APS SMA': df['APS SMA'],
    'APS KULIAH': df['APS KULIAH']
}

def random_forest_model(y, nama_target):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"HASIL RANDOM FOREST - {nama_target}")
    print("="*50)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    importance = pd.DataFrame({
        'Variabel': X.columns,
        'Importance': model.feature_importances_
    })

    importance = importance.sort_values(
        by='Importance',
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance)

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Data Aktual")
    plt.ylabel("Data Prediksi")

    plt.title(f"Random Forest {nama_target}")

    plt.grid(True)

    target_key = nama_target.lower().replace(" ", "_")
    plt.savefig(
        FIGURE_DIR / f"random_forest_{target_key}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return model

model_sd = random_forest_model(targets['APS SD'], 'APS SD')

model_smp = random_forest_model(targets['APS SMP'], 'APS SMP')

model_sma = random_forest_model(targets['APS SMA'], 'APS SMA')

model_kuliah = random_forest_model(targets['APS KULIAH'], 'APS KULIAH')

print("\n" + "="*50)
print("FORECASTING APS 2026-2030")
print("="*50)

aps_tahunan = df.groupby('Tahun')[[
    'APS SD',
    'APS SMP',
    'APS SMA',
    'APS KULIAH'
]].mean()

tahun = np.array(aps_tahunan.index)

tahun_prediksi = np.array([2026, 2027, 2028, 2029, 2030])

def linear_trend_forecast(data, nama):

    coef = np.polyfit(tahun, data, 1)

    trend = np.poly1d(coef)

    prediksi = trend(tahun_prediksi)

    hasil = pd.DataFrame({
        'Tahun': tahun_prediksi,
        'Prediksi APS': prediksi
    })

    print("\n" + "="*50)
    print(f"PREDIKSI APS 2026-2030 : {nama}")
    print("="*50)

    print(hasil)

    plt.figure(figsize=(8,5))

    plt.plot(
        tahun,
        data,
        marker='o',
        label='Data Aktual'
    )

    plt.plot(
        tahun_prediksi,
        prediksi,
        marker='o',
        linestyle='--',
        label='Forecast'
    )

    plt.title(f"Forecasting {nama}")

    plt.xlabel("Tahun")
    plt.ylabel("APS")

    plt.legend()
    plt.grid(True)

    target_key = nama.lower().replace(" ", "_")
    plt.savefig(
        FIGURE_DIR / f"forecast_{target_key}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return hasil

forecast_sd = linear_trend_forecast(
    aps_tahunan['APS SD'],
    'APS SD'
)

forecast_smp = linear_trend_forecast(
    aps_tahunan['APS SMP'],
    'APS SMP'
)

forecast_sma = linear_trend_forecast(
    aps_tahunan['APS SMA'],
    'APS SMA'
)

forecast_kuliah = linear_trend_forecast(
    aps_tahunan['APS KULIAH'],
    'APS KULIAH'
)

with pd.ExcelWriter(OUTPUT_DIR / "aps_forecasts_2026_2030.xlsx") as writer:

    forecast_sd.to_excel(
        writer,
        sheet_name='APS_SD',
        index=False
    )

    forecast_smp.to_excel(
        writer,
        sheet_name='APS_SMP',
        index=False
    )

    forecast_sma.to_excel(
        writer,
        sheet_name='APS_SMA',
        index=False
    )

    forecast_kuliah.to_excel(
        writer,
        sheet_name='APS_KULIAH',
        index=False
    )
print(f"\nHasil forecast tersimpan di: {OUTPUT_DIR / 'aps_forecasts_2026_2030.xlsx'}")
print(f"Visualisasi tersimpan di: {FIGURE_DIR}")