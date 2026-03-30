"""
evaluator.py
============
Post-training evaluation:
  - Computes MAE and RMSE on the test set (in original AQI scale)
  - Saves a predicted vs actual plot to outputs/plots/
  - Returns a metrics dict so train.py can print a clean summary
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PLOT_DIR


def evaluate_model(model, X_test, y_test, target_scaler=None):
    """
    Evaluate model on the held-out test set.

    Parameters
    ----------
    model          : Trained tf.keras.Model
    X_test         : np.ndarray
    y_test         : np.ndarray (normalized)
    target_scaler  : fitted MinMaxScaler for AQI (to inverse-transform)

    Returns
    -------
    metrics : dict with 'MAE' and 'RMSE'
    y_pred  : np.ndarray of predicted AQI values (original scale)
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    y_pred_norm = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform back to real AQI scale
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred = y_pred_norm
        y_true = y_test

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3)}
    print(f"\n[evaluator] Test MAE : {metrics['MAE']}")
    print(f"[evaluator] Test RMSE: {metrics['RMSE']}")

    _plot_predictions(y_true, y_pred)
    return metrics, y_pred


def _plot_predictions(y_true, y_pred, n_points: int = 200):
    """Save a predicted vs actual line chart (last n_points for clarity)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true[-n_points:], label="Actual AQI",    color="#2196F3", linewidth=1.5)
    ax.plot(y_pred[-n_points:], label="Predicted AQI", color="#FF5722",
            linewidth=1.5, linestyle="--")
    ax.set_title("AirGuard — Predicted vs Actual AQI (Test Set)", fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[evaluator] Plot saved to: {plot_path}")
