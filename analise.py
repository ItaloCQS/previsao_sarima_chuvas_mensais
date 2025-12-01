import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error

!pip install pmdarima
from pmdarima import auto_arima

# 1) CARREGAR DADOS E LIMPAR " --- "
df = pd.read_csv("data/chuvas.csv")

# Remove espaços dos nomes das colunas
df.columns = df.columns.str.strip()

# Trocar "---" por NaN
df = df.replace("---", np.nan)

# Converter colunas numéricas
for col in df.columns:
    if col != "Ano":
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Nulos antes do preenchimento:")
print(df.isna().sum())

# 2) TRANSFORMAR PARA FORMATO LONGO
df_long = df.melt(id_vars="Ano", var_name="Mes", value_name="Chuva")

# Converter meses para número
mapa_meses = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
    "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}
df_long["Mes_num"] = df_long["Mes"].map(mapa_meses)

# Criar coluna de data
df_long["Data"] = pd.to_datetime(df_long["Ano"].astype(str) + "-" + df_long["Mes_num"].astype(str) + "-01")

# 3) PREENCHER NULOS PELA MÉDIA CLIMATOLÓGICA DO MÊS
media_mes = df_long.groupby("Mes_num")["Chuva"].transform("mean")
df_long["Chuva"] = df_long["Chuva"].fillna(media_mes)

print("Nulos depois do preenchimento:")
print(df_long["Chuva"].isna().sum())

# 4) CRIAR SÉRIE TEMPORAL
series = df_long.set_index("Data")["Chuva"].sort_index()

print(series.head())

# 5) ANÁLISE BÁSICA
print(series.describe())
plt.figure(figsize=(12,4))
plt.plot(series)
plt.title("Série Temporal - Chuva Mensal")
plt.show()

# 6) MÉDIA MÓVEL, ACF E PACF
plt.figure(figsize=(12,4))
plt.plot(series.rolling(12).mean())
plt.title("Média Móvel de 12 meses")
plt.show()

plot_acf(series, lags=40)
plt.show()

plot_pacf(series, lags=40)
plt.show()

# 7) DECOMPOSIÇÃO
decomp = seasonal_decompose(series, model="additive", period=12)
decomp.plot()
plt.show()

# 8) TESTE DE ESTACIONARIEDADE E NORMALIDADE

adf = adfuller(series)
print("ADF:", adf[0])
print("p-value:", adf[1])

# Normalidade
stat, p = shapiro(series)
print("Shapiro-Wilk p-value:", p)

# Se não estacionária, diferenciar
if adf[1] > 0.05:
    series_diff = series.diff().dropna()
else:
    series_diff = series

# Treino até os últimos 12 meses
train = series[:-12]
test = series[-12:]

# SARIMA AUTOMÁTICO
model = auto_arima(
    train,
    seasonal=True,
    m=12,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

print(model.summary())

# PREVISÃO
forecast = model.predict(n_periods=12)
forecast = pd.Series(forecast, index=test.index)

plt.figure(figsize=(12,4))
plt.plot(series, label="Histórico")
plt.plot(forecast, label="Previsão SARIMA", linewidth=3)
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(series, label="Histórico")
plt.plot(forecast, label="Previsão SARIMA", linewidth=3)

start_zoom = series.index[-48]

plt.xlim(start_zoom, forecast.index[-1])
plt.legend()
plt.show()

"""Dados de teste para avaliar a previsão"""

# 1) TREINO E TESTE
train = series[:-12]
test = series[-12:]

# 2) AJUSTAR MODELO SARIMA AUTOMÁTICO

model = auto_arima(
    train,
    seasonal=True,
    m=12,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

# 3) PREVISÃO PARA O PERÍODO DE TESTE

forecast = model.predict(n_periods=12)
forecast = pd.Series(forecast, index=test.index)

# 4) MÉTRICAS DE AVALIAÇÃO

rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("📊 Avaliação do Modelo SARIMA:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 5) GRÁFICO — PREVISÃO VS REAL (com zoom no final)

plt.figure(figsize=(12,4))
plt.plot(series, label="Histórico")
plt.plot(test, label="Teste (Real)", linewidth=3)
plt.plot(forecast, label="Previsão SARIMA", linewidth=3)

inicio_zoom = test.index[0] - pd.DateOffset(months=24)
fim_zoom = test.index[-1]

plt.figure(figsize=(12,4))
plt.plot(series, label="Histórico")
plt.plot(test, label="Real (Teste)", linewidth=3)
plt.plot(forecast, label="Previsão", linewidth=3)

plt.xlim(inicio_zoom, fim_zoom)
plt.legend()
plt.grid(True)
plt.show()