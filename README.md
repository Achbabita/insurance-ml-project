# Insurance ML Project

## Overview
This project analyzes an insurance dataset using a complete machine learning workflow.  
The work includes:
- Data cleaning and preparation
- Feature scaling of numeric variables (`age`, `bmi`, `children`) with StandardScaler
- Dummy variable encoding for categorical features (`region`, `bmi_category`)
- Creation of quartile bins for charges (`charges_bin`) using `pd.qcut`
- Correlation analysis (Pearson) to identify relationships between features and charges
- Chi-Square tests of independence to evaluate associations between categorical features and charge bins
- Train/test split of the dataset using scikit-learn’s train_test_split for reproducible model evaluation
- Model training with Linear Regression
- Model evaluation using 𝑅2, adjusted 𝑅2, and residual analysis
- Visualization of feature relationships and model performance
  
The project delivers a fully processed dataset with statistical insights, ready for modeling and further analysis.


## Dataset
The dataset includes demographic and lifestyle features:
- Age
- Gender (is_female)
- BMI
- Children
- Smoker status (is_smoker)
- Region (dummy variables)
- BMI category (dummy variables)
- Charges (target variable)

## Preprocessing
- **Scaling:** Standardized numeric features (`age`, `bmi`, `children`) using `StandardScaler`.
- **Encoding:** Applied dummy variable encoding for categorical features (`region`, `bmi_category`) with `drop_first=True` to avoid the dummy variable trap.
- **Binning:** Created `charges_bin` using `pd.qcut()` to split charges into quartiles for classification tasks.
- Splitting: Used train_test_split(X, Y, test_size=0.33, random_state=42) to divide the dataset into training (67%) and testing (33%) sets with reproducibility.

## Statistical Analysis
- **Correlation Analysis (Pearson):**
  - Strong correlation: `is_smoker` with charges (~0.79).
  - Moderate correlation: `age` (~0.30), `bmi` (~0.20).
  - Weak/negative correlations: region and BMI category dummies.
- **Chi-Square Test of Independence:**
  - Tested categorical features against `charges_bin`.
  - Decision rule: Reject null (keep feature) if `p < 0.05`, otherwise drop feature.

## Modeling & Evaluation
- Model: Linear Regression trained on the processed dataset.
- Prediction: Generated predictions on the test set.
- Evaluation Metrics:
     - 𝑅2 score: ~0.804
     - Adjusted 𝑅2 score: ~0.799
     - Residual plots confirm model assumptions and highlight variance patterns.

## Tech Stack
- Python
- pandas
- scikit-learn
- scipy
- seaborn / matplotlib
- Jupyter Notebook

