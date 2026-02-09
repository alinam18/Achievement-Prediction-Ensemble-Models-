# Achievement-Prediction-Ensemble-Models-
Predicting Early Reading and Math Outcomes Using Interpretable Ensemble Learning

## Project Overview
This project investigates the key drivers of reading and mathematics performance among kindergarten students. Using a multi-level dataset (student, teacher, and school levels), we employed a variety of econometric and machine learning models—ranging from **OLS** and **Regularized Linear Regression** to **XGBoost** and **Neural Networks**—to identify the most accurate predictors of academic success.

## Key Findings
* **Top Model:** **XGBoost** outperformed all other models with a combined Mean Squared Error (MSE) of **990.22**.
* **Primary Determinant:** **Socioeconomic Status (SES)**, proxied by eligibility for free/reduced-price lunch, was the strongest predictor of academic outcomes.
* **Environment Matters:** Students in **rural areas** showed higher proficiency in reading, while those in **suburban areas** excelled in mathematics.
* **Class Size:** Consistent with **Project STAR** findings, smaller class sizes had a notable positive impact on performance.

## Tech Stack & Methodology
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `XGBoost`, `LightGBM`, `SHAP`, `matplotlib`, `seaborn`
* **Data Size:** 5,060 observations with 14 variables.
* **Preprocessing Pipeline:** * Median imputation for numerical data.
    * 'Unknown' category imputation for categorical data.
    * One-hot encoding & category collapsing.
    * 80/20 Train-Test split.

## Model Performance Comparison
We evaluated several models based on **MSE** to determine the best fit for capturing non-linear relationships in educational data.

| Model Type | Performance Notes |
| :--- | :--- |
| **OLS** | Suffered from multicollinearity among predictors. |
| **Ridge/LASSO** | Improved accuracy by penalizing redundant features. |
| **Decision Tree** | Highly prone to overfitting. |
| **Random Forest** | Significantly reduced variance/overfitting compared to a single tree. |
| **XGBoost** | **Winner.** Best at capturing multidimensional interactions. |
| **Neural Network** | Underperformed due to small dataset size and sparse input. |

## Feature Importance & Insights
To move beyond "black-box" predictions, we utilized **SHAP (SHapley Additive exPlanations)** to interpret the XGBoost model.



### 1. Socioeconomic Influence
The data reveals that lunch status (SES) is the most critical driver. The model identified that interventions in class size and school quality are most impactful for students from low-SES backgrounds.

### 2. Geographic Divergence
The learning environment interacts differently with subjects:
* **Reading:** Stronger performance observed in **Rural** settings.
* **Math:** Stronger performance observed in **Suburban** settings.

### 3. Classroom Dynamics
The analysis confirmed that **small class sizes** correlate with higher scores, validating long-standing educational theories regarding early childhood intervention.

---

## How to Run
1. Clone the repository
2. Run the notebook on this repo
