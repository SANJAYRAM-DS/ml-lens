# 📘 **math4ml — A Lightweight Math & Stats Library for Machine Learning**

**math4ml** is a modular, NumPy-backed Python library designed to **teach**, **visualize**, and **compute** the mathematics behind AI & Machine Learning.

It combines:

- **Linear algebra**
- **Statistics**
- **Probability**
- **Hypothesis testing**
- **Preprocessing**
- **Educational examples**

---

# 🚀 **Features**

## **1. Linear Algebra**

- Matrix operations: `matmul`, `add`, `subtract`, `transpose`, `inverse`, `det`, …
- Vector operations: `dot`, `norm`, `angle`, `projection`, …
- Decompositions: **LU**, **QR**, **SVD** *(optional upgrade)*

---

## **2. Statistics**

- Descriptive stats: `mean`, `var`, `std`, `median`, `range`
- Correlation: **Pearson**, **Spearman**
- Distributions: **normal**, **binomial**, **uniform**, **Poisson**
- Hypothesis tests:
  - **t-test**
  - **chi-square test**
  - **ANOVA**
  - **z-test**
  - **non-parametric tests** (coming soon)

---

## **3. Probability**

- PMF, PDF, CDF utilities
- Combinatorics: `nCr`, `nPr`
- Bayes theorem helpers
- Random variable simulation utilities

---

## **4. Preprocessing**

- Scaling:
  - StandardScaler
  - MinMaxScaler
  - MaxAbsScaler
  - RobustScaler
- Encoding:
  - One-hot
  - Label
  - Binary
- Feature engineering helpers

---
## **5.optimization**

-

## **6.ml_models**
-classification_models
    -"LogisticRegression",
    -"NaiveBayes",
    -"KNN"
-linear_models
    -"LinearRegression",
    -"RidgeRegression",
    -"LassoRegression"
-metrics
    -"RegressionMetrics",
    -"ClassificationMetrics"
-validation
    -"CrossValidation"

## **7. Educational Tools**

Every function includes:

- 🧮 **Mathematical formula**
- 📘 **Concept explanation**
- 🔍 **Assumptions**
- ✏️ **Step-by-step example**
- 📓 **Jupyter notebook tutorials**

Perfect for **students learning ML math**, **data scientists**, and **AI researchers**.

---

# 📦 **Installation**

### **PyPI**

```bash
pip install math4ml


**🧠 Quickstart Example**
just use print(math4ml.linalg.__doc__), print(math4ml.__doc__) or help(math4ml)

from math4ml.linalg import matmul
from math4ml.stats import t_test

print(matmul([[1, 2]], [[3], [4]]))

stat, p = t_test([1,2,3], [3,4,5])
print("T-stat:", stat, "P-value:", p)

📚 Tutorials
🔍 Explore: https://github.com/SANJAYRAM-DS/math4ml.tutorials.git

Contains:

-Linear algebra examples

-Statistical tests

-Probability examples

-Preprocessing tutorials

-optimization

-ml_models

🤝 Contributing

We welcome contributions from everyone!

You can help by:

-🐛 Reporting issues

-🌟 Suggesting features

-📘 Improving documentation

-🧪 Adding tests

-🧩 Adding examples

-🔧 Submitting pull requests

📝 License

MIT License — free for commercial, educational, and research use.
