# 📱 Smartphone Price Predictor

> A machine learning pipeline that predicts smartphone prices from raw technical specifications — featuring automated feature engineering, outlier-robust preprocessing, and a serialized production-ready model.

---

## 🚀 Project Overview

Smartphone pricing is influenced by a complex interplay of hardware specs, brand positioning, and market segmentation. This project builds an end-to-end regression pipeline that:

- **Parses messy, real-world product listings** into structured numeric features using regex-based extraction
- **Engineers meaningful features** (RAM, storage, battery capacity, fast-charging wattage, display size, refresh rate, camera resolution, brand)
- **Trains a Gradient Boosting Regressor** with log-transformed targets to handle the skewed price distribution
- **Serializes the pipeline** for seamless deployment and inference on new data

---

## 📊 Results

| Metric | Score |
|---|---|
| R² Score | **~0.90+** |
| Mean Absolute Error | Low MAE on INR price scale |
| Target Transformation | `log1p` → `expm1` for stable regression |

---

## 🧠 Technical Highlights

### Feature Engineering from Raw Text
Rather than relying on pre-cleaned data, the pipeline extracts structured signals directly from raw product description strings:

```python
df['ram']          = df['memory'].str.extract(r'(\d+)\?GB RAM').astype(float)
df['battery_mAh']  = df['battery'].str.extract(r'(\d+)\?mAh').astype(float)
df['charging_watt']= df['battery'].str.extract(r'(\d+)W').astype(float)
df['refresh_rate'] = df['display'].str.extract(r'(\d+) Hz').astype(float)
```

### Outlier Removal with IQR Capping
A custom IQR-based filter removes extreme price outliers using a configurable spread factor, preserving data integrity without hard-coding thresholds:

```python
def remove_outliers_iqr(df, col='price', factor=2):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return df[(df[col] >= low) & (df[col] <= high)].copy()
```

### Sklearn Pipeline Architecture
The full preprocessing and modeling workflow is encapsulated in a single `sklearn.Pipeline`:

```
ColumnTransformer
├── OneHotEncoder         → brand (categorical)
└── Pipeline
    ├── SimpleImputer     → median imputation for missing specs
    └── StandardScaler    → feature normalization
        │
GradientBoostingRegressor (n_estimators=1200, max_depth=8, lr=0.1)
```

This design ensures **zero data leakage**, clean train/test separation, and a single serializable object for deployment.

---

## 🗂️ Features Used

| Feature | Source Column | Extraction Method |
|---|---|---|
| `brand` | `name` | First token, lowercased |
| `ram` | `memory` | Regex: `(\d+) GB RAM` |
| `storage` | `memory` | Regex: `(\d+) GB inbuilt` |
| `battery_mAh` | `battery` | Regex: `(\d+) mAh` |
| `charging_watt` | `battery` | Regex: `(\d+)W` |
| `display_size` | `display` | Regex: `(\d+.\d+) inches` |
| `refresh_rate` | `display` | Regex: `(\d+) Hz` |
| `rear_camera` | `camera` | Regex: `(\d+) MP` |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numeric-lightblue?logo=numpy)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blueviolet)

- **pandas** — data loading and manipulation
- **numpy** — numerical operations and log-transform
- **scikit-learn** — preprocessing, pipeline, and GBM model
- **seaborn / matplotlib** — EDA and price distribution visualization
- **joblib** — model serialization

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smartphone-price-predictor.git
cd smartphone-price-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

### 3. Run the notebook
```bash
jupyter notebook phone.ipynb
```

### 4. Predict on new data
After training, load the serialized pipeline and predict instantly:

```python
import joblib
import pandas as pd
import numpy as np

model = joblib.load("phone_price_predictor.pkl")

new_phone = pd.DataFrame({
    "brand":        ["samsung"],
    "ram":          [8],
    "storage":      [256],
    "battery_mAh":  [5000],
    "charging_watt":[25],
    "display_size": [6.62],
    "refresh_rate": [120],
    "rear_camera":  [50]
})

predicted_price = np.expm1(model.predict(new_phone))
print(f"Predicted Price: ₹{predicted_price[0]:,.0f}")
```

---

## 📁 Project Structure

```
smartphone-price-predictor/
│
├── phone.ipynb                  # Main notebook (EDA + training)
├── data.csv                     # Raw smartphone listings dataset
├── phone_price_predictor.pkl    # Serialized model pipeline
└── README.md
```

---

## 📈 EDA Insights

- Price distribution is **right-skewed** → log transformation (`log1p`) applied to stabilize variance
- **Brand** is a strong categorical signal — captured via one-hot encoding with rare brand grouping (`Others` for brands with ≤11 listings)
- Box plots reveal significant price variance within brands, confirming that hardware specs carry meaningful predictive signal *beyond* brand alone

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] Add processor/chipset as a feature (via NLP on spec strings)
- [ ] Deploy as a REST API using FastAPI + Docker
- [ ] Build an interactive price estimator with Streamlit

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
