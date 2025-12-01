# 🌧️ Previsão de Chuva Mensal com Séries Temporais (SARIMA)

Este repositório contém uma análise completa de **chuva mensal histórica** com técnicas de 
**séries temporais**, incluindo:

- Limpeza e tratamento da base
- Conversão para série temporal
- Análise exploratória
- Decomposição
- ACF e PACF
- Teste de estacionaridade
- Transformações e diferenciação
- Modelo SARIMA
- Avaliação com dados de teste
- Previsão futura

---

## 📌 Objetivo

Criar um modelo estatístico capaz de prever os valores de chuva mensal utilizando 
técnicas clássicas de séries temporais, com foco em interpretação e robustez.

## 🧹 1. Pré-processamento

- Padronização das colunas
- Conversão de vírgula para ponto
- Conversão de valores inválidos em `NaN`
- Preenchimento com média mensal
- Criação da série temporal

---

## 📊 2. Análise Exploratória e Diagnóstica

Inclui:

- Série temporal original  
- Média móvel  
- ACF e PACF  
- Decomposição (tendência, sazonalidade e resíduo)  
- Teste ADF (estacionaridade)  

---

## 🤖 3. Modelagem — SARIMA

O modelo foi escolhido automaticamente usando `auto_arima`:

```python
model = auto_arima(
    train,
    seasonal=True,
    m=12,
    stepwise=True
)
