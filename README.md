<h1 align="center">🌴 Holiday Package Prediction 🌴</h1>
<h3 align="center">With Random Forest Classification</h3>

<p align="center">
  <img src="images/holiday_package_prediction.gif" alt="Holiday Package Prediction" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Random%20Forest-Classifier-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Accuracy-93%25-brightgreen?style=for-the-badge" />
</p>

---

## 📌 Project Overview

**Trips & Travel.Com** wants to expand its customer base by launching a new **Wellness Tourism Package** — defined as travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle and sense of well-being.

Previously, customers were contacted **randomly** for marketing without using any available data — leading to high marketing costs and a low conversion rate of only **18%**.

This project builds a **Random Forest Classification model** to predict which customers are likely to purchase the Wellness Tourism Package, so the company can **target the right customers** and make marketing expenditure more efficient.

---

## 📁 Repository Structure

```
Holiday-Package-Prediction-Randomforest/
├── images/
│   ├── holiday_package_prediction.gif
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── auc.png
│
├── Holiday_Package_Prediction.ipynb
├── Travel.csv
├── holiday_package_classification_model.pkl
├── preprocessor.pkl
├── .gitignore
└── README.md
```

---

## 🛠️ Libraries Used

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`

---

## 📊 Dataset

The dataset is sourced from [Kaggle — Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)

- **Samples:** 4,888
- **Features:** 20
- **Target Variable:** `ProdTaken` — 1 (Purchased) / 0 (Not Purchased)
- **Purchase Rate:** ~18% (class imbalance present)

| Feature | Description |
|---|---|
| Age | Age of the customer |
| MonthlyIncome | Monthly income of the customer |
| DurationOfPitch | Duration of the sales pitch (in minutes) |
| Passport | Whether the customer has a passport (1/0) |
| NumberOfTrips | Number of trips taken by the customer |
| PitchSatisfactionScore | Customer satisfaction score for the pitch |
| NumberOfFollowups | Number of followups made by the agent |
| TotalVisiting | Total number of people visiting (person + children) |
| CityTier | Tier of the city the customer belongs to |
| Occupation | Customer's occupation |

---

## ⚙️ Workflow

1. Load the dataset from `Travel.csv`
2. Handle missing values — median for continuous, mode for discrete features
3. Fix categorical inconsistencies (`Fe Male` → `Female`, `Single` → `Unmarried`)
4. Drop irrelevant feature — `CustomerID`
5. Feature Engineering — create `TotalVisiting` from `NumberOfPersonVisiting` + `NumberOfChildrenVisiting`
6. Exploratory Data Analysis — distributions and correlation heatmap
7. Train-Test Split — 80% train / 20% test (`random_state=42`)
8. Preprocessing — `OneHotEncoder` for categorical, `StandardScaler` for numerical via `ColumnTransformer`
9. Train and compare multiple models — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
10. Hyperparameter tuning using `RandomizedSearchCV` with 100 iterations and 3-fold CV
11. Evaluate using Confusion Matrix, Classification Report, and ROC-AUC Curve
12. Feature Importance analysis
13. Save model and preprocessor using `joblib`

---

## 📈 Exploratory Data Analysis

<p align="center">
  <img src="images/feature_distributions.png" alt="Feature Distributions" width="48%"/>
  &nbsp;
  <img src="images/correlation_heatmap.png" alt="Correlation Heatmap" width="48%"/>
</p>

**Key Insights:**
- `Passport` has the strongest positive correlation with purchase (0.26)
- `Age` and `MonthlyIncome` are highly correlated with each other (0.46)
- `DurationOfPitch` and `NumberOfFollowups` positively influence purchase decision
- Most customers are from **City Tier 1** and prefer **3-star properties**

---

## 🤖 Model Comparison

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | ~81% | ~81% |
| Decision Tree | ~100% | ~85% |
| Gradient Boosting | ~93% | ~90% |
| **Random Forest** | **~100%** | **~93%** |

Random Forest was selected as the final model based on best test accuracy and F1 score.

---

## 🔧 Hyperparameter Tuning

```python
rf_params = {
    "max_depth"        : [5, 8, 15, None, 10],
    "max_features"     : [5, 7, "sqrt", 8],
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators"     : [100, 200, 500, 1000]
}
```

Tuning method: `RandomizedSearchCV` — `n_iter=100`, `cv=3`, `n_jobs=-1`

**Best Parameters:**
```python
RandomForestClassifier(
    n_estimators=1000,
    min_samples_split=2,
    max_features=7,
    max_depth=None
)
```

---

## 📉 Model Results

<p align="center">
  <img src="images/confusion_matrix.png" alt="Confusion Matrix" width="48%"/>
  &nbsp;
  <img src="images/feature_importance.png" alt="Feature Importance" width="48%"/>
</p>

**Classification Report:**
```
               precision    recall  f1-score   support

Not Purchased     0.93      0.99      0.96       787
    Purchased     0.97      0.68      0.80       191

     accuracy                         0.93       978
    macro avg     0.95      0.84      0.88       978
 weighted avg     0.94      0.93      0.93       978
```

**Top Predictive Features:**
1. 💰 MonthlyIncome
2. 🎂 Age
3. ⏱️ DurationOfPitch
4. 🛂 Passport
5. ✈️ NumberOfTrips

---

## 🧪 Sample Prediction

```python
import joblib
import pandas as pd

model = joblib.load('holiday_package_classification_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# New customer sample
sample = pd.DataFrame({
    'TypeofContact'         : ['Self Enquiry'],
    'CityTier'              : [1],
    'DurationOfPitch'       : [20],
    'Gender'                : ['Male'],
    'NumberOfFollowups'     : [4],
    'ProductPitched'        : ['Deluxe'],
    'PreferredPropertyStar' : [3],
    'MaritalStatus'         : ['Unmarried'],
    'NumberOfTrips'         : [3],
    'Passport'              : [1],
    'PitchSatisfactionScore': [4],
    'OwnCar'                : [1],
    'Occupation'            : ['Salaried'],
    'MonthlyIncome'         : [25000],
    'Age'                   : [30],
    'Designation'           : ['Executive'],
    'TotalVisiting'         : [3],
})

sample_transformed = preprocessor.transform(sample)
prediction = model.predict(sample_transformed)
probability = model.predict_proba(sample_transformed)

print("Prediction  :", "✅ Will Purchase" if prediction[0] == 1 else "❌ Will Not Purchase")
print("Probability :", f"Not Purchased = {probability[0][0]*100:.1f}%  |  Purchased = {probability[0][1]*100:.1f}%")
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AnmolPatel20/Holiday-Package-Prediction-Randomforest.git
cd Holiday-Package-Prediction-Randomforest
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**3. Run the notebook**
```bash
jupyter notebook Holiday_Package_Prediction.ipynb
```

---

## 📌 Notes
- Both `holiday_package_classification_model.pkl` and `preprocessor.pkl` must be loaded together for prediction
- The preprocessor handles all encoding and scaling — never pass raw data directly to the model

---

## 🙋 About
I'm on my machine learning journey — building, experimenting and documenting as I go. Every notebook here represents something I've genuinely tried to understand, not just run. 🚀

- GitHub: [@AnmolPatel20](https://github.com/AnmolPatel20)
- Portfolio: [anmolpatel20.github.io/Anmol_Portfolio](https://anmolpatel20.github.io/Anmol_Portfolio/)

## 🙏 Acknowledgements
Thanks to **Krish Naik Sir** whose Udemy course has been a great resource throughout this learning journey.

*"The best way to learn is to do."*

---

<p align="center">⭐ Star this repo if you found it helpful!</p>
