# 💼 Insurance Cost Analysis & Prediction

This project explores the key factors affecting individual medical insurance charges and develops predictive models to estimate these costs. We use **regression modeling**, **feature engineering**, and **regularization** techniques to ensure reliable and interpretable predictions.

## 📊 Objective

To understand what drives insurance costs (e.g., smoking habits, age, BMI), and to build models that can accurately predict insurance charges using Python and machine learning.

## 🧰 Tools & Libraries

- Python (Jupyter Notebook)
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LinearRegression, PolynomialFeatures, Ridge, GridSearchCV, Pipeline)

## 🧪 Project Workflow

### 1. Data Cleaning & Preprocessing
- Handled missing values:
  - `age` filled with **mean**
  - `smoker` filled with **mode**
- Encoded categorical variables: `sex`, `smoker`, `region`
- Feature scaling applied where necessary

### 2. Exploratory Data Analysis (EDA)
- Identified strong correlations:
  - **Smoker** has the strongest impact on cost
  - **Age** and **BMI** are also significant predictors
- Visualized relationships using scatter plots, box plots, and heatmaps

### 3. Modeling

| Model                    | Description                            | R² Performance |
|-------------------------|----------------------------------------|----------------|
| Simple Linear Regression| Baseline model with numeric features   | Moderate       |
| Polynomial Regression   | Captured non-linear relationships      | Improved       |
| Ridge Regression        | Reduced overfitting via regularization | Best           |

- Used **Pipeline** to combine scaling + polynomial feature transformation + modeling
- Applied **GridSearchCV** to optimize hyperparameter `alpha` for Ridge Regression

### 4. Evaluation
- Compared models using **R² scores** on train and test sets
- Analyzed **underfitting / overfitting** behavior
- Final Ridge Regression model achieved the most balanced generalization

---

## 🎯 Key Findings

- **Smokers pay significantly more** for insurance (~double or more).
- **Polynomial regression** captures curvilinear relationships (e.g., age vs. cost).
- **Ridge regression** with optimized alpha prevented overfitting and improved robustness.

---

## 📌 Future Improvements

- Add interaction terms between features (e.g., age × smoker)
- Try additional regularization methods (e.g., Lasso)
- Use more advanced models like **Random Forests** or **XGBoost**

---

## 👨‍💻 Author

Mert Ovet | [GitHub Profile](https://github.com/your-profile)  
Project developed as part of personal data science portfolio

