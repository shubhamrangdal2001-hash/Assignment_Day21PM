"""
Part B — Bias-Variance Tradeoff
Week 04 · Day 21 · PM Session
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# B1 — Generate Noisy Dataset
# ─────────────────────────────────────────────────────────

# True underlying function: y = sin(x) + noise
n = 40
X = np.linspace(0, 3 * np.pi, n)
y = np.sin(X) + np.random.normal(0, 0.4, n)

X_plot = np.linspace(0, 3 * np.pi, 300)

# ─────────────────────────────────────────────────────────
# B2 — Fit Polynomial Models of Increasing Complexity
# ─────────────────────────────────────────────────────────

degrees = [1, 2, 5, 10, 15]
train_errors = []

fig, axes = plt.subplots(1, len(degrees), figsize=(18, 4))
fig.suptitle("Bias-Variance Tradeoff: Polynomial Models", fontsize=13)

for i, deg in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X.reshape(-1, 1), y)

    y_train_pred = model.predict(X.reshape(-1, 1))
    train_mse = mean_squared_error(y, y_train_pred)
    train_errors.append(train_mse)

    y_plot = model.predict(X_plot.reshape(-1, 1))

    axes[i].scatter(X, y, color='steelblue', s=20, label='Data', alpha=0.7)
    axes[i].plot(X_plot, y_plot, color='red', label=f'Degree {deg}')
    axes[i].plot(X_plot, np.sin(X_plot), color='green', linestyle='--', alpha=0.5, label='True fn')
    axes[i].set_title(f'Degree {deg}\nMSE={train_mse:.3f}')
    axes[i].set_ylim(-3, 3)
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('part_b_models.png', dpi=120)
plt.show()
print("Plot saved: part_b_models.png")

# ─────────────────────────────────────────────────────────
# B3 — Training Error vs Model Complexity
# ─────────────────────────────────────────────────────────

plt.figure(figsize=(7, 4))
plt.plot(degrees, train_errors, marker='o', color='crimson', label='Training Error')
plt.xlabel('Polynomial Degree (Model Complexity)')
plt.ylabel('Training MSE')
plt.title('Training Error vs Model Complexity')
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
plt.savefig('part_b_error_curve.png', dpi=120)
plt.show()
print("Plot saved: part_b_error_curve.png")

# ─────────────────────────────────────────────────────────
# B4 — Print Explanation
# ─────────────────────────────────────────────────────────

print("\nB — Bias-Variance Tradeoff Explanation")
print("=" * 55)
print("""
Bias:
  - Error from wrong assumptions in the model.
  - A high-bias model is too simple — it misses patterns.
  - Example: degree-1 line trying to fit a sine wave (underfitting).

Variance:
  - Error from the model being too sensitive to training data.
  - A high-variance model learns noise, not patterns.
  - Example: degree-15 polynomial that wiggles everywhere (overfitting).

Optimal Model:
  - Sits in the middle — low enough complexity to avoid overfitting,
    but complex enough to capture real patterns.
  - In this example, degree 3-5 gives a good balance.
""")

print("Train MSEs by degree:")
for deg, err in zip(degrees, train_errors):
    label = "← underfitting" if deg == 1 else ("← overfitting" if deg >= 10 else "")
    print(f"  Degree {deg:>2}: MSE = {err:.4f}  {label}")
