"""
Part D — AI-Augmented Task
Week 04 · Day 21 · PM Session
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar

Prompt used:
    "Explain regression vs classification and bias-variance tradeoff
     with Python examples and visualizations."

This file documents the AI output and my evaluation of it.
"""

# ─────────────────────────────────────────────────────────
# AI-Generated Python Examples (from prompt response)
# I ran and verified these myself before including them here.
# ─────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# ── Regression Example (from AI) ──────────────────────────
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([1.5, 3.1, 4.8, 6.2, 8.1, 9.5, 11.0, 12.8])

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)

print("=== AI Example: Regression ===")
print(f"Coefficients: {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print("Verified: Correct — linear model fits the data well.\n")

# ── Classification Example (from AI) ─────────────────────
X_c = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y_c = np.array([0, 0, 0, 0, 1, 1, 1, 1])

clf = LogisticRegression().fit(X_c, y_c)
accuracy = clf.score(X_c, y_c)
print("=== AI Example: Classification ===")
print(f"Accuracy: {accuracy:.2f}")
print("Verified: Correct — logistic regression classifies binary labels.\n")

# ── Bias-Variance Visualization (from AI) ─────────────────
X_bv = np.linspace(0, 10, 50)
y_bv = np.sin(X_bv) + np.random.normal(0, 0.5, 50)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = ["Degree 1 (Underfitting)", "Degree 4 (Good Fit)", "Degree 12 (Overfitting)"]
degrees_d = [1, 4, 12]
X_plot = np.linspace(0, 10, 300)

for ax, deg, title in zip(axes, degrees_d, titles):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_bv.reshape(-1, 1), y_bv)
    ax.scatter(X_bv, y_bv, s=15, color='steelblue', alpha=0.6)
    ax.plot(X_plot, model.predict(X_plot.reshape(-1, 1)), color='red')
    ax.set_title(title)
    ax.set_ylim(-3, 3)

plt.suptitle("Part D — AI-Generated Bias-Variance Plot (Verified)", fontsize=12)
plt.tight_layout()
plt.savefig('part_d_ai_visualization.png', dpi=120)
plt.show()
print("Plot saved: part_d_ai_visualization.png")

# ─────────────────────────────────────────────────────────
# Evaluation of AI Output
# ─────────────────────────────────────────────────────────
print("""
=== Evaluation of AI Output ===

1. Are the explanations correct?
   Yes. The AI correctly explained:
   - Regression outputs continuous values; classification outputs labels.
   - Bias = underfitting (model too simple).
   - Variance = overfitting (model too complex).
   - The optimal model balances the two.

2. Do the visualizations correctly show underfitting and overfitting?
   Yes, after I ran the code:
   - Degree 1 (underfitting): straight line misses the sine curve.
   - Degree 4 (good fit): follows the pattern without over-wiggling.
   - Degree 12 (overfitting): fits training data but is wildly unstable.

3. Any corrections needed?
   - The AI's initial classification example used a continuous feature
     that was not normalized. I kept it as-is since sklearn handles it,
     but in a real project I'd standardize the features first.
   - Overall, the AI output was accurate and the visuals were correct.
""")
