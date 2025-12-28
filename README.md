# 🩺 Diabetes Prediction System (Machine Learning)

## 📌 Project Overview

This project is a **machine learning–based diabetes risk prediction system** designed to predict whether a **woman** is likely to have diabetes based on medical parameters.
The model is trained using historical health data and applies **K-Nearest Neighbors (KNN)** classification to make predictions for new patients.

⚠️ **Disclaimer:**
This project is for **educational purposes only** and should **not be used as a medical diagnosis tool**.

---

## 🎯 Problem Statement

Diabetes is a chronic disease that requires early detection to reduce long-term health risks.
The goal of this project is to build a machine learning model that can **predict diabetes risk** using commonly available medical features.

---

## 📊 Dataset Description

* Dataset: **PIMA Indians Diabetes Dataset**
* Total Records: **768**
* Target Variable: `Outcome`

  * `0` → No Diabetes
  * `1` → Diabetes

### ⚠️ Important Note

* The dataset contains **only female patients**
* Feature **`Pregnancies`** makes this model **women-specific**
* Some medical features contain **zero values**, which represent missing data

### Features Used

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skin fold thickness  |
| Insulin                  | Serum insulin                |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Genetic diabetes risk        |
| Age                      | Age in years                 |
| Outcome                  | Diabetes result (Target)     |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and quality of the dataset.

Techniques used:

* **Boxplots** → Detect outliers
* **Histograms** → Understand data distribution
* **Pairplots** → Analyze relationships between features
* **Correlation Heatmap** → Identify important predictors

Key insights:

* Glucose and BMI show strong correlation with diabetes
* Dataset contains outliers and encoded missing values
* Class imbalance exists between diabetic and non-diabetic patients

---

## ⚙️ Data Preprocessing

Steps performed:

1. Removed target column (`Outcome`) from feature set
2. Applied **StandardScaler** to normalize features
3. Split data into **training (70%)** and **testing (30%)**
4. Avoided data leakage by fitting scaler only on training data

---

## 🤖 Machine Learning Model

* Algorithm Used: **K-Nearest Neighbors (KNN)**
* Reason for choosing KNN:

  * Works well for non-linear data
  * Simple and interpretable
* Hyperparameter tuning performed by testing multiple `K` values
* Final model selected based on **test accuracy**

---

## 🧠 How Prediction Works

For a new woman:

1. User enters medical parameters
2. Input is scaled using the same scaler used during training
3. KNN finds the **K nearest patients**
4. Majority voting determines the prediction
5. Output indicates **diabetes risk**

---

## 🧪 Example Prediction

```python
new_patient = [[
    1, 140, 80, 30, 100, 32.5, 0.45, 35
]]

new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
```

Output:

* `1` → High chance of Diabetes
* `0` → Low chance of Diabetes

---

## 🚀 Future Improvements

* Handle missing values more realistically (replace zeros with median)
* Compare multiple models (Logistic Regression, Random Forest)
* Improve recall (important for healthcare use cases)
* Deploy model using **Streamlit** or **Flask**
* Extend dataset to include male patients

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

## 👨‍💻 Author

**Veman S Chippa**
📍 India
🔗 LinkedIn: [https://www.linkedin.com/in/veman-chippa/](https://www.linkedin.com/in/veman-chippa/)

---

## ⭐ How to Use This Repo

1. Clone the repository
2. Open the notebook in Google Colab or Jupyter
3. Run cells step-by-step
4. Modify input values to test new predictions
