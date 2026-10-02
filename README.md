# Indian AQI Streamlit Dashboard

A Streamlit dashboard for analysis and prediction of Air Quality Index
(AQI) for Indian cities.

## Features

### Analysis

- City selection
- Year selection
- AQI trend
- Seasonal AQI analysis
- AQI category distribution

### Prediction

- Pollution parameter input
- Weather parameter input
- Time parameters
- Location parameters
- Random Forest AQI prediction

## Machine Learning

The AQI prediction model uses:

- Random Forest Regressor
- Mutual Information feature selection
- Numerical preprocessing
- Categorical preprocessing
- One-hot encoding

### Selected Features

- PM2.5
- PM10
- PM Ratio
- CO
- NO2
- SO2
- O3
- Dust
- AOD
- Dew Point
- Is Raining
- Solar Radiation
- Is Daytime
- Latitude
- Longitude
- Month
- Hour
- Time of Day

## Model Performance

RMSE: 54.65

MAE: 16.80

R²: 0.763

## Installation

```bash
pip install -r requirements.txt
```
## Running File
```bash
streamlit run app.py
```
or
```bash
py -m streamlit run app.py
```


This app requires Data and Model:
- INDIA_AQI_COMPLETE_20251126.csv
- aqi_random_forest_pipeline.pkl
