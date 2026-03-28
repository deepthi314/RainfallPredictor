# Manila Rainfall Prediction Project

This project predicts daily rainfall (`PRECTOTCORR`) for Manila, Philippines using meteorological data from NASA POWER (1995-2025).

## Project Structure
- `data/`: Raw and processed datasets.
- `src/`: Core Python modules for preprocessing, feature engineering, PCA, training, and prediction.
- `models/`: Saved best model and scaler.
- `app/`: Streamlit web application.
- `notebooks/`: Interactive Jupyter notebooks for each step.
- `outputs/`: Figures and model comparison reports.

## Installation
1. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
2. Download data (if not present):
   Data is fetched from NASA POWER for coordinates 14.5822°N, 120.9751°E.

## Usage
### Running the Pipeline
Run the modules in order or use the pipeline script:
```bash
python src/preprocessing.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

### Launching the Dashboard
```bash
streamlit run app/app.py
```

## Features
- **Data Cleaning**: Handles missing values (-999) using monthly median imputation.
- **Feature Engineering**: Incorporates time features (Month, Season), Lags (1, 7, 30 days), and Rolling Means.
- **PCA**: Analyzes feature dimensionality and explains variance.
- **Models**: Compares Decision Tree, Random Forest, XGBoost, SVR, and MLP Neural Networks.
- **Evaluation**: Comprehensive metrics (R2, RMSE, MAE) and visualizations.
