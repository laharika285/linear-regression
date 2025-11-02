 **TASK 1: Simple Linear Regression (Sales vs Advertising)**.
## 🧠 TASK 1 — Simple Linear Regression (Sales vs Advertising)

### 📘 Project Overview

This project demonstrates how to build a **Simple Linear Regression** model using **Scikit-learn** to study the **linear relationship between Sales and Advertising** expenditure for a **dietary weight control product**.

The objective is to understand how changes in advertising spending influence sales, using a simple predictive model.

---

### 📈 What is Linear Regression?

**Linear Regression** is a supervised learning algorithm used to model the relationship between a **dependent variable (Y)** and an **independent variable (X)** by fitting a straight line to the data.

The mathematical form is:

[
Y = b_0 + b_1X
]

Where:

* **Y** = Dependent variable (Sales)
* **X** = Independent variable (Advertising)
* **b₀** = Intercept
* **b₁** = Slope or coefficient

Scikit-learn’s `LinearRegression()` automatically computes these values using the **least squares method**.

---

### ⚙️ Implementation Steps

1. **Import Required Libraries**

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn import metrics
   ```

2. **Load and Explore the Dataset**

   * Load dataset using Pandas.
   * Display first few records and summary statistics.
   * Visualize the relationship between `Advertising` and `Sales`.

3. **Split Dataset**

   * Split the data into **training (80%)** and **testing (20%)** sets using:

     ```python
     train_test_split(X, y, test_size=0.2, random_state=42)
     ```

4. **Train the Model**

   * Fit a **LinearRegression()** model to the training data.
   * Obtain intercept and coefficient values.

5. **Make Predictions**

   * Predict sales on the test set using the trained model.

6. **Evaluate the Model**

   * Compute metrics like:

     * Mean Absolute Error (MAE)
     * Mean Squared Error (MSE)
     * Root Mean Squared Error (RMSE)
     * R² Score
   * Visualize:

     * Regression line
     * Residuals plot

---

### 📊 Example Evaluation Metrics

| Metric | Meaning                 | Example Value |
| ------ | ----------------------- | ------------- |
| MAE    | Average absolute error  | 1.24          |
| MSE    | Mean squared error      | 2.18          |
| RMSE   | Root mean squared error | 1.48          |
| R²     | Model fit quality       | 0.87          |

---

### 🧩 File Structure

```
📂 SimpleLinearRegression/
├── advertising.csv           # Dataset (Advertising vs Sales)
├── simple_linear_regression_advertising.py  # Python script
├── README.md                 # Project documentation
└── requirements.txt          # Required libraries
```

---

### 📦 Installation & Requirements

**Install dependencies:**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**Run the project:**

```bash
python simple_linear_regression_advertising.py
```

If you don’t have a dataset file, the script automatically generates synthetic data for demonstration.

---

### 📉 Visualization Output

1. **Scatter Plot (Sales vs Advertising)**
   Shows the relationship between the two variables.

2. **Regression Line Plot**
   Displays how well the fitted line represents the data trend.

3. **Residual Plot**
   Helps check for randomness of errors (good fit → no clear pattern).

---

### 🧠 Conclusion

* The **Linear Regression model** successfully predicts sales based on advertising expenditure.
* A positive coefficient indicates that higher advertising spending leads to increased sales.
* The model performance can be further improved by adding more features (e.g., TV, social media, digital marketing spend).

---

### 👨‍💻 Author

**Name:** K.laharika
**Course:** Computer Science (cyber security)
**Topic:** Machine Learning — Simple Linear Regression using Scikit-learn

---

Would you like me to create a short **`requirements.txt`** file and a **GitHub-ready folder structure** (with example data and script names) to go along with this README?
