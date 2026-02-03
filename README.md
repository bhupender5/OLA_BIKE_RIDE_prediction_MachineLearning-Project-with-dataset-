#  OLA Bike Ride Demand Prediction using Machine Learning

Predicting hourly OLA bike ride demand using real-world historical data with Regression models and feature engineering.

---

## 📌 Project Overview

This project focuses on forecasting **bike ride counts** based on:

* Date & Time
* Season
* Weather conditions
* Temperature
* Humidity
* Wind speed
* Casual & Registered users

The goal is to help ride-sharing platforms like **OLA** optimize:

✅ Fleet allocation
✅ Driver availability
✅ Demand forecasting
✅ Operational efficiency

---

## 🗂 Dataset Information

* Total rows: **10,886**
* Features: **14**
* Target variable: **count (number of rides)**

### Key Features

| Feature   | Description               |
| --------- | ------------------------- |
| season    | Season of the year        |
| weather   | Weather condition         |
| temp      | Temperature               |
| humidity  | Humidity level            |
| windspeed | Wind speed                |
| time      | Hour of day               |
| day       | Day of month              |
| month     | Month                     |
| weekday   | Weekday/Weekend           |
| am_or_pm  | Morning/Evening indicator |

---

## ⚙️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn

---

## 🔧 Workflow

### 1️⃣ Data Preprocessing

* Missing value handling (Forward Fill)
* Datetime feature extraction
* Feature engineering (hour, day, month, weekday, am/pm)
* Standard Scaling

### 2️⃣ Exploratory Data Analysis

* Demand vs time trends
* Seasonal effects
* Weather impact
* Distribution plots

### 3️⃣ Model Training

Implemented:

* Linear Regression
* Support Vector Machine
* Random Forest Regressor

### 4️⃣ Evaluation Metrics

Used:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)

---

## 📊 Model Performance

| Model             | MAE     | RMSE    |
| ----------------- | ------- | ------- |
| Linear Regression | 28.62   | 36.01   |
| SVM               | 37.48   | 46.62   |
| Random Forest     | ⭐ 25.44 | ⭐ 31.57 |

### ✅ Best Model → Random Forest Regressor

Random Forest achieved the **lowest error**, making it the best choice for demand prediction.

---

## 🚀 How to Run

### Clone repo

```bash
git clone https://github.com/bhupender5/OLA_BIKE_RIDE_prediction_MachineLearning-Project-with-dataset-
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run notebook

```bash
jupyter notebook
```

---

## 📈 Sample Code

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

model = RandomForestRegressor()
model.fit(X_train, Y_train)

pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(Y_val, pred))
print("RMSE:", rmse)
```

---

## 💡 Key Learnings

* Feature engineering improves accuracy significantly
* Time-based features are critical for demand forecasting
* Tree-based models outperform linear models
* Scaling is important for SVM

---

## 🔮 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* XGBoost / LightGBM
* Deep Learning (LSTM for time-series)
* Deployment using Flask/Streamlit
* Real-time dashboard

---

## 👨‍💻 Author

**Bhupender Singh**
Data Analytics | Machine Learning | Data Science

GitHub: [https://github.com/bhupender5](https://github.com/bhupender5)
LinkedIn: [https://www.linkedin.com/in/bhupinder-singh-bba271187](https://www.linkedin.com/in/bhupinder-singh-bba271187)

---

⭐ If you found this helpful, don’t forget to star the repo!
