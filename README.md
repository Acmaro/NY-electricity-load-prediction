# NY Day-ahead Electricity Load Forecasting

This project focuses on predicting short-term electricity load demand in New York State using machine learning models such as LSTM and GRU. Accurate load forecasting helps improve grid stability and energy efficiency.

## Context

Electricity demand forecasting is a time-series problem affected by factors such as weather, calendar events, and economic activity. The goal is to compare several machine learning models and identify the one that best predicts day-ahead or week-ahead electricity load.

## Dataset

### NYISO Load Data
- Frequency: 12 points/hour (288/day)
- Source: [NYISO Load Data](https://www.nyiso.com/load-data)

### Historical Weather Data
- Frequency: 1 point/hour (24/day)
- Source: [Weather Underground API](https://www.wunderground.com/history)

### Features
- Load data
- Weather attributes (temp, humidity, etc.)
- Calendar features (holiday/workday/weekend)

## Preprocessing

- Timestamp conversion and alignment
- Feature reduction based on correlation
- Data augmentation with workday flag
- Imputation and normalization
- Custom scaling for inverse transform on predictions

### Sampling Strategy

| Scenario | Training Period | Validation Period | Testing Period |
|----------|------------------|--------------------|----------------|
| 1. Pre-Pandemic dataset | 2015–2019         | –                | 2020 |
| 2. Recent dataset   | 2019–2021         | 2022             | 2023 |


### RNN Architecture
- Input size: 8 features
- Output: 288 (daily load)
- Hidden Layers: 2 (size: 32–128, tuned)
- Dropout: 0.3 (tuned)
- Optimizer: Adam with learning rate scheduling
- Epochs: 30
- Loss: MSE
  
## Hyperparameter Tuning (on 2019-2022 dataset)

Used Ray Tune to explore:
- Batch size
- Dropout rate
- Hidden size
- Learning rate

Example of best GRU model (2022 tuning):
- Batch size: 64
- Dropout: 0.2782
- Hidden size: 32
- Learning rate: 1.88e-5

RNN (LSTM) Model fine-tuning using Ray Tune (ordered by validation loss):
| Batch Size | Drop Rate | Hidden Size | Learning Rate | Train Loss | Validation Loss |
| ---------- | --------- | ----------- | ------------- | ---------- | --------------- |
| 256        | 0.0318    | 128         | 0.0001255     | 0.02075    | 0.1274          |
| 128        | 0.2954    | 64          | 0.0008027     | 0.08143    | 0.1384          |
| 64         | 0.3550    | 128         | 0.0008593     | 0.06722    | 0.1385          |
| 64         | 0.3901    | 16          | 0.0000476     | 0.11654    | 0.1424          |
| 128        | 0.3695    | 16          | 0.0002582     | 0.11775    | 0.1453          |
| 64         | 0.3429    | 32          | 0.0002147     | 0.04496    | 0.1615          |

RNN (GRU) Model fine-tuning using Ray Tune (ordered by validation loss):
| Batch Size | Drop Rate | Hidden Size | Learning Rate | Train Loss | Validation Loss |
| ---------- | --------- | ----------- | ------------- | ---------- | --------------- |
| 64         | 0.2782    | 32          | 1.8888e-05    | 0.06248    | 0.0863          |
| 256        | 0.3915    | 32          | 1.3182e-04    | 0.05852    | 0.0893          |
| 128        | 0.4761    | 32          | 3.7783e-04    | 0.05358    | 0.1112          |
| 64         | 0.1675    | 64          | 7.8110e-04    | 0.02694    | 0.1690          |

## Evaluation

Metrics:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

### Results on 2015–2019 Training / 2020 Testing Dataset:

| Model        |R²    |
|--------------|------|
| LSTM (RNN)   | 0.843 |
| GRU (RNN)    | 0.855 |


### Results: 2019–2021 Training / 2022–2023 Testing

| Year | Model | MAE   | MAPE | R²    |
|------|--------|--------|------|--------|
| 2022 | GRU    | 75.62 | 5.65 | 0.827 |
| 2023 | GRU    | 71.80 | 5.80 | 0.792 |

### NYISO Benchmarks:

| Year | MAE   | MAPE | R²    |
|------|--------|------|------|
| 2022 | 55.08 | 4.05 | 0.919 |
| 2023 | 52.52 | 4.15 | 0.900 |


## Limitations

- RNN models are prone to overfitting.
- COVID-19 significantly impacted electricity patterns, reducing model generalization.
- Public policy influences on demand are difficult to predict and model.

## References

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Islam et al. (2022), MPE - Electrical Load Forecasting](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2316474)
- [NYISO Load Trends and COVID-19 Impact](https://www.nyiso.com/documents/20142/2347703/2022-Climate-Impact-Study.pdf)
