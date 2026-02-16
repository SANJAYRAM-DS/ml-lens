<div align="center">

<img src="img.png" alt="ML-Lense Logo" width="200"/>

# ML-Lense

### Understand Machine Learning — Not Just Use It.

Explainable, interpretable, and educational tooling that reveals *how ML really works* under the hood.

[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

</div>

---

## ✨ What is ML-Lense?

**ML-Lense** is an open-source library that helps you **see inside machine learning**.

Instead of treating ML as a black box, ML-Lense explains:

- Why a model made a prediction  
- How parameters were learned  
- What math drives each algorithm  
- How features influence outcomes  
- What happens during training step-by-step  

It is designed for:

- Researchers  
- Students  
- Engineers  
- Educators  
- Curious builders  

If scikit-learn helps you **build models**,  
ML-Lense helps you **understand them**.

---

## 🧠 Philosophy

Modern ML tooling focuses on:

- Accuracy
- Speed
- Scale

But rarely on:

- Understanding
- Interpretability
- First-principles learning

ML-Lense exists to change that.

> ML should be understandable, not mysterious.

---

## 🔍 Core Idea

Train any model → Get a **deep explanation layer**.

```python
from sklearn.linear_model import LinearRegression
from ml_lense import explain

model = LinearRegression().fit(X, y)
report = explain(model, X, y)

report.show()
```

ML-Lense generates:

- Parameter reasoning
- Mathematical derivations
- Feature importance breakdown
- Training dynamics
- Human-readable explanations

---

## 🚀 Features

### 1️⃣ Parameter Transparency
Understand why weights were chosen.

- Linear models → closed-form math
- Neural nets → gradient dynamics
- Tree models → split logic

### 2️⃣ Math-Aware Explanations
Every output maps back to math.

- Loss function analysis
- Gradient flow visualization
- Optimization reasoning
- Statistical assumptions

### 3️⃣ Training Dynamics
See what happens during learning.

- Convergence behavior
- Bias vs variance shifts
- Overfitting detection
- Learning stability

### 4️⃣ Feature Attribution
Not just importance — causation-aware insights

- Local explanations
- Global influence
- Interaction effects
- Feature sensitivity curves

### 5️⃣ First-Principles Reports
ML-Lense produces structured outputs:

- Markdown reports
- PDF summaries
- Interactive notebooks
- Teaching-ready visualizations

---

## 📚 Supported Domains (Roadmap)

### Classical ML
- Linear Regression
- Logistic Regression
- SVM
- kNN
- Decision Trees
- Random Forest
- Gradient Boosting

### Deep Learning
- MLPs
- CNNs
- Transformers (planned)
- Attention analysis

### Statistics
- Hypothesis testing insights
- t-tests, chi-square reasoning
- Distribution assumptions
- Sampling diagnostics

### Time Series
- ARIMA interpretability
- Seasonality decomposition
- Forecast uncertainty analysis

---

## 🧪 Example Outputs

ML-Lense produces explanations like:

> "Weight w₃ is large because feature variance is high and strongly correlated with the target."

> "Model confidence comes from low residual spread across 92% of samples."

> "Gradient oscillations indicate suboptimal learning rate."

---

## 🎯 Use Cases

### 🎓 Education
- Learn ML intuitively
- Teach algorithms visually
- Bridge math ↔ implementation

### 🔬 Research
- Interpret experimental models
- Debug training failures
- Generate explanation reports

### 🏭 Production ML
- Model debugging
- Stakeholder explainability
- Responsible AI documentation

---

## 🆚 How is this different?

| Tool | Focus |
|------|-------|
| scikit-learn | Model building |
| SHAP/LIME | Feature attribution |
| TensorBoard | Training metrics |
| **ML-Lense** | **Full-stack understanding** |

ML-Lense combines:

- Math
- Training behavior
- Interpretability
- Intuition

In one unified layer.

---

## � Installation

```bash
pip install ml-lense
```

Or development install:

```bash
git clone https://github.com/SANJAYRAM-DS/ml-lense
cd ml-lense
pip install -e .
```

---

## ⚡ Quick Start

```python
from ml_lense import explain
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
model = Ridge().fit(X, y)

lens = explain(model, X, y)

lens.summary()
lens.math()
lens.training_dynamics()
lens.feature_story()
```

---

## 📊 Outputs

ML-Lense can generate:

- 📄 Markdown reports
- 📘 Research-ready PDFs
- 📊 Visual dashboards
- 🧠 Math walkthroughs

---

## 🧱 Design Principles

- Minimal API
- Model-agnostic
- Math-first
- Research-grade clarity
- Documentation-friendly

---

## 🤝 Contributing

We welcome contributors from:

- ML researchers
- Students
- Educators
- Open-source developers

Ways to contribute:

- Add model explainers
- Improve math modules
- Create tutorials
- Expand domains (time series, NLP, CV)

See CONTRIBUTING.md for details.

---

## 🌍 Vision

ML-Lense aims to become:

**The "NumPy for understanding machine learning."**

A universal layer that makes every model:

- Transparent
- Learnable
- Explainable
- Trustworthy

---

## � License

MIT License — use freely in research and industry.

---

## ⭐ If you like ML-Lense

- Star the repo
- Share with learners
- Use in teaching
- Contribute ideas

**Let's make machine learning understandable.**

---

<div align="center">

Built with clarity in mind.

</div>