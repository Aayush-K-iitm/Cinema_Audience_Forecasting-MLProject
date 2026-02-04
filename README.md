# Cinema Audience Forecasting (Kaggle Competition)

**Competition**: [Cinema_Audience_Forecasting_challenge](https://www.kaggle.com/competitions/Cinema_Audience_Forecasting_challenge)

Graduate project predicting daily theatre audience counts using booking data, calendar features, and time-series lags.

## Competition Results
- **My best Public LB**: ~0.37 (LightGBM + log1p target + booking lags)
- **Top score**: 0.5
- **Key insight**: Boosting generalizes better than RF here despite RF's strong local CV

## My Best Notebook
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/aayushkonar/23f2000866-notebook-t32025?scriptVersionId=276285871)

## Models & Approach
| Model | Local RMSE | Public LB | Notes |
|-------|------------|-----------|-------|
| RF + Classifier | 17.7 | ~0.10-0.15 | Overfits validation |
| LightGBM Ensemble | ~18.5 | **0.37** | **Best** - log target + lags |
| Statistical Avg | - | 0.351 | Simple 7-feature blend |

## Files
- `03_lightgbm_final.ipynb` - Best LB model (random+grid search)
- `02_rf_model.ipynb` - RF exploration (shows overfitting lesson)
- `01_data_prep.ipynb` - Feature engineering pipeline
- `submission_final.csv` - Best submission

## Key Learnings
- Log1p target + booking lags beats RF
- TimeSeriesSplit CV essential for time data
- Tree models robust to correlated features (dow/is_weekend)
