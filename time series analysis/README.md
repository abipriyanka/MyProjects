# Time Series Analysis on Champagne Sales Data

## Introduction

This project involves the analysis of champagne sales data to understand its trends, patterns, and seasonality. It includes exploratory data analysis, stationarity testing, and the development of ARIMA and seasonal ARIMA models for sales prediction.

## Key Inferences

- Sales peaked in 1969 and 1971, with all years showing good sales records.
- December recorded the highest sales, with November also showing a spike, while August had the least sales.
- Seasonal decomposition revealed an increasing trend in sales with seasonality.
- Stationarity tests indicated non-stationarity, but after seasonal differencing, the series became stationary.

## Model Building

### ARIMA Model:
- An ARIMA model was built with order (1, 1, 1), but the forecast was not efficient due to the model not considering seasonality.

### Seasonal ARIMA Model:
- A seasonal ARIMA model with order (1, 1, 1) and seasonal order (1, 1, 1, 12) was built, resulting in a more accurate forecast.

