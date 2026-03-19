# Week 04 · Day 21 (PM Session) — Assignment

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**  
**Topics:** Regression, Classification, Bias–Variance Tradeoff
Gitlink:  https://github.com/shubhamrangdal2001-hash/Assignment_Day21PM.git
---

## Overview

This assignment covers:
- Creating synthetic datasets for regression and classification
- Training and evaluating linear/logistic regression models
- Manually implementing MSE and accuracy
- Simulating the bias–variance tradeoff with polynomial models
- Evaluating AI-generated explanations and code

---

## Folder Structure

```
.
├── part_a_regression_classification.py   # Part A — all 6 tasks
├── part_b_bias_variance.py               # Part B — stretch problem
├── part_c_interview.py                   # Part C — interview Qs + calculate_mse
├── part_d_ai_task.py                     # Part D — AI-augmented task
└── README.md
```

---

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib scikit-learn
```

### Run each part

```bash
# Part A
python part_a_regression_classification.py

# Part B
python part_b_bias_variance.py

# Part C
python part_c_interview.py

# Part D
python part_d_ai_task.py
```

Each script prints outputs to the terminal and saves plots as `.png` files.

---

## Output Files Generated

| File | Description |
|------|-------------|
| `part_a_models.png` | Linear and logistic regression plots |
| `part_b_models.png` | Polynomial fits of increasing degree |
| `part_b_error_curve.png` | Training error vs model complexity |
| `part_d_ai_visualization.png` | AI-generated bias-variance visualization |

---

## Key Concepts

| Concept | One-liner |
|---------|-----------|
| Regression | Predicts a continuous number (e.g., house price) |
| Classification | Predicts a category (e.g., spam/not spam) |
| Bias | Error from being too simple — underfitting |
| Variance | Error from being too complex — overfitting |
| MSE | Mean of squared differences between actual and predicted |
