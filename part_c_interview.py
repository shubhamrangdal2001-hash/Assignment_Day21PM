"""
Part C — Interview Ready
Week 04 · Day 21 · PM Session
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
"""

import numpy as np

# ─────────────────────────────────────────────────────────
# Q2 — Coding: calculate_mse function
# ─────────────────────────────────────────────────────────

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error manually.

    Parameters:
        y_true : list or array — actual target values
        y_pred : list or array — predicted values

    Returns:
        float — MSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    errors = y_true - y_pred
    mse = np.mean(errors ** 2)
    return mse


# ─────────────────────────────────────────────────────────
# Test the function
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test case 1
    y_true_1 = [3, -0.5, 2, 7]
    y_pred_1 = [2.5, 0.0, 2, 8]
    print("Test 1")
    print(f"  y_true: {y_true_1}")
    print(f"  y_pred: {y_pred_1}")
    print(f"  MSE   : {calculate_mse(y_true_1, y_pred_1):.4f}")

    # Test case 2 — perfect predictions
    y_true_2 = [1, 2, 3, 4, 5]
    y_pred_2 = [1, 2, 3, 4, 5]
    print("\nTest 2 — perfect predictions")
    print(f"  y_true: {y_true_2}")
    print(f"  y_pred: {y_pred_2}")
    print(f"  MSE   : {calculate_mse(y_true_2, y_pred_2):.4f}")

    # Test case 3
    y_true_3 = [10, 20, 30]
    y_pred_3 = [12, 18, 35]
    print("\nTest 3")
    print(f"  y_true: {y_true_3}")
    print(f"  y_pred: {y_pred_3}")
    print(f"  MSE   : {calculate_mse(y_true_3, y_pred_3):.4f}")

    # ─────────────────────────────────────────────────────
    # Q1 — Regression vs Classification answers (printed)
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q1 — Difference between Regression and Classification")
    print("=" * 60)
    print("""
Regression predicts a continuous numeric value.
Classification predicts a discrete class label.

Real-world examples:
  Regression      → Predicting house price, stock price, temperature
  Classification  → Email spam detection (spam/not spam),
                    Disease diagnosis (positive/negative),
                    Image recognition (cat/dog/car)
""")

    print("=" * 60)
    print("Q3 — Bias-Variance Tradeoff")
    print("=" * 60)
    print("""
Bias:
  Bias is the error that comes from a model being too simple.
  A high-bias model makes strong (wrong) assumptions about data
  and misses the underlying patterns — this is called underfitting.

  Example: Fitting a straight line to data that follows a curve.

Variance:
  Variance is the error from the model being too sensitive to
  the training data. It learns the noise along with the signal.
  This is called overfitting.

  Example: A degree-15 polynomial that passes through every
  training point but performs badly on new data.

Underfitting (High Bias, Low Variance):
  - Model is too simple.
  - Both training and test errors are high.
  - Solution: Use a more complex model.

Overfitting (Low Bias, High Variance):
  - Model is too complex.
  - Training error is very low, test error is high.
  - Solution: Regularization, more data, or simpler model.

Optimal Model:
  - Balances bias and variance.
  - Low training error AND low test error.
  - Generalizes well to unseen data.
""")
