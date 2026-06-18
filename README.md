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

## Statistical Analysis
- **Correlation Analysis (Pearson):**
  - Strong correlation: `is_smoker` with charges (~0.79).
  - Moderate correlation: `age` (~0.30), `bmi` (~0.20).
  - Weak/negative correlations: region and BMI category dummies.
- **Chi-Square Test of Independence:**
  - Tested categorical features against `charges_bin`.
  - Decision rule: Reject null (keep feature) if `p < 0.05`, otherwise drop feature.

## Tech Stack
- Python
- pandas
- scikit-learn
- scipy
- seaborn / matplotlib
- Jupyter Notebook

