"""
Part A — Regression and Classification
Week 04 · Day 21 · PM Session
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# A1 — Synthetic Datasets
# ─────────────────────────────────────────────────────────

# Regression dataset (continuous target)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Classification dataset (binary target)
X_clf, y_clf = make_classification(
    n_samples=100, n_features=1, n_informative=1,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# ─────────────────────────────────────────────────────────
# A2 — Train Models and Visualize
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Linear Regression ---
lin_model = LinearRegression()
lin_model.fit(X_reg, y_reg)
y_reg_pred = lin_model.predict(X_reg)

axes[0].scatter(X_reg, y_reg, color='steelblue', label='Actual', alpha=0.6)
axes[0].plot(X_reg, y_reg_pred, color='red', label='Predicted line')
axes[0].set_title('Linear Regression')
axes[0].set_xlabel('Feature')
axes[0].set_ylabel('Target (continuous)')
axes[0].legend()

# --- Logistic Regression ---
log_model = LogisticRegression()
log_model.fit(X_clf, y_clf)
y_clf_pred = log_model.predict(X_clf)

axes[1].scatter(X_clf, y_clf, color='orange', alpha=0.6, label='Actual class')
# Probability curve
x_range = np.linspace(X_clf.min(), X_clf.max(), 300).reshape(-1, 1)
prob = log_model.predict_proba(x_range)[:, 1]
axes[1].plot(x_range, prob, color='darkgreen', label='P(class=1)')
axes[1].axhline(0.5, linestyle='--', color='gray', label='Threshold 0.5')
axes[1].set_title('Logistic Regression')
axes[1].set_xlabel('Feature')
axes[1].set_ylabel('Probability / Class')
axes[1].legend()

plt.tight_layout()
plt.savefig('part_a_models.png', dpi=120)
plt.show()
print("Plot saved: part_a_models.png")

# ─────────────────────────────────────────────────────────
# A3 — Identify Problem Type from Target Variable
# ─────────────────────────────────────────────────────────

def identify_problem(target):
    unique_vals = np.unique(target)
    if len(unique_vals) <= 10 and set(unique_vals).issubset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
        return "Classification — target has discrete class labels"
    else:
        return "Regression — target is continuous"

print("\nA3 — Problem Type Identification")
print("Regression dataset target:", identify_problem(y_reg))
print("Classification dataset target:", identify_problem(y_clf))

# ─────────────────────────────────────────────────────────
# A4 — Manual Linear Regression + MSE
# ─────────────────────────────────────────────────────────

print("\nA4 — Manual Linear Regression")

# Small dataset
x_manual = np.array([1, 2, 3, 4, 5], dtype=float)
y_manual = np.array([2.2, 4.1, 5.8, 8.3, 10.1])

# Slope and intercept using least squares formula
n = len(x_manual)
m = (n * np.sum(x_manual * y_manual) - np.sum(x_manual) * np.sum(y_manual)) / \
    (n * np.sum(x_manual ** 2) - np.sum(x_manual) ** 2)
b = (np.sum(y_manual) - m * np.sum(x_manual)) / n

print(f"Slope (m): {m:.4f}, Intercept (b): {b:.4f}")

# Predict using linear equation
y_pred_manual = m * x_manual + b
print("Predicted values:", np.round(y_pred_manual, 2))

# Manual MSE
mse_manual = np.mean((y_manual - y_pred_manual) ** 2)
print(f"MSE (manual): {mse_manual:.4f}")

# Verify with sklearn
mse_sklearn = mean_squared_error(y_manual, y_pred_manual)
print(f"MSE (sklearn check): {mse_sklearn:.4f}")

# ─────────────────────────────────────────────────────────
# A5 — Simple Classification with Threshold + Accuracy
# ─────────────────────────────────────────────────────────

print("\nA5 — Manual Classification with Threshold")

# Continuous scores (e.g., exam marks out of 100)
scores = np.array([45, 72, 30, 88, 55, 61, 40, 95, 50, 78])
true_labels = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

threshold = 60
predicted_labels = (scores >= threshold).astype(int)
print("Scores:", scores)
print("True labels:", true_labels)
print("Predicted labels:", predicted_labels)

# Manual accuracy
correct = np.sum(predicted_labels == true_labels)
accuracy = correct / len(true_labels)
print(f"Correct predictions: {correct}/{len(true_labels)}")
print(f"Accuracy (manual): {accuracy:.2f}")
print(f"Accuracy (sklearn check): {accuracy_score(true_labels, predicted_labels):.2f}")

# ─────────────────────────────────────────────────────────
# A6 — Regression vs Classification Comparison
# ─────────────────────────────────────────────────────────

print("\nA6 — Regression vs Classification Comparison")
print("-" * 50)
comparison = {
    "Aspect"          : ["Type of Output", "Use Cases", "Evaluation Metrics"],
    "Regression"      : ["Continuous (e.g., 12.5, 98.7)", "Price prediction, temp forecast", "MSE, RMSE, R²"],
    "Classification"  : ["Discrete class labels (0/1, cat/dog)", "Spam detection, disease diagnosis", "Accuracy, Precision, Recall, F1"],
}
for i in range(3):
    print(f"{comparison['Aspect'][i]:<22} | Regression: {comparison['Regression'][i]:<35} | Classification: {comparison['Classification'][i]}")
