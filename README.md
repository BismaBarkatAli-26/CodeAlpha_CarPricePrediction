# 🚗 Car Price Prediction with Machine Learning — CodeAlpha Data Science Internship (Task 3)

## Project description
A regression project that predicts the selling price of used cars (in lakhs ₹) from their age, showroom price when new, kilometres driven, fuel type, seller type, transmission and number of previous owners.

## Objectives
- Clean the data and engineer useful features (car age)
- Handle numeric and categorical variables with a leakage-safe preprocessing pipeline
- Compare five regression models with cross-validation
- Evaluate with MAE, MSE, RMSE and R²; visualise actual vs predicted prices
- Explain which factors the model associates with price

## Dataset
- **Source:** Kaggle — "Vehicle dataset" (CarDekho): https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho — file **`car data.csv`**
- **Size:** ≈301 cars × 9 columns
- **Columns:** `Car_Name`, `Year`, `Selling_Price` (**target**, lakhs ₹), `Present_Price` (new-car showroom price, lakhs ₹), `Driven_kms` (or `Kms_Driven`), `Fuel_Type`, `Selling_type` (or `Seller_Type`), `Transmission`, `Owner`
- Note: the dataset has no horsepower column; `Present_Price` serves as a proxy for brand/segment value.

## Technologies
Python 3 · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter / Google Colab

## Project structure
```
CodeAlpha_CarPricePrediction/
├── Car_Price_Prediction.ipynb
├── requirements.txt
├── README.md
├── data/      # put car data.csv here
├── images/    # charts saved by the notebook
└── models/    # car_price_model.joblib (created when you run the notebook)
```

## How to run
1. Download `car data.csv` from Kaggle.
2. **Colab:** upload notebook + CSV, Runtime → Run all.
   **Local:** `pip install -r requirements.txt`, copy the CSV into `data/`, run `jupyter notebook Car_Price_Prediction.ipynb`.

## Methodology
1. Standardise column names across dataset versions; validate required columns
2. Clean: numeric conversion, drop duplicates / invalid prices
3. Feature engineering: `Car_Age`; drop high-cardinality `Car_Name`
4. EDA: price distribution (skew), scatter/box plots, correlation heatmap
5. 80/20 train-test split; `ColumnTransformer` (StandardScaler + OneHotEncoder) inside a `Pipeline`
6. Compare Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting via 5-fold CV (R², RMSE, MAE)
7. Train best model; evaluate on test set; actual-vs-predicted and residual plots
8. Permutation importance + linear coefficients; save model; prediction function

## Results

| Model | CV R² | CV RMSE |
|---|---|---|
| Linear Regression |0.8850|1.6451|
| Ridge Regression |0.8858|1.6384|
| Decision Tree |0.7419|2.1300|
| Random Forest |0.8930|1.5099|
| Gradient Boosting |0.8783|1.6237|

**Best model:** `Random Forest` — Test MAE `1.4968`, MSE `12.8684`, RMSE `3.5873`, R² `0.5007` (6.11 lakhs ₹)

Screenshots: `images/05_model_comparison.png`, `images/06_actual_vs_predicted.png`, `images/08_feature_importance.png`

## Limitations
- Small dataset; a few expensive cars strongly affect RMSE
- Missing features (horsepower, engine, condition, location, brand)
- Asking prices from one website/time period

## Future improvements
Log-transform the target, `GridSearchCV` tuning, use `Car details v3.csv` (same Kaggle dataset) for engine/power/mileage features, Streamlit deployment.

## Author
Bisma Barkat Ali — Data Science Intern
