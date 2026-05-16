import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def print_metrics(metrics):
    print("\n--- Evaluation Metrics ---")
    print(f"  RMSE : ${metrics['RMSE']:>12,.0f}")
    print(f"  MAE  : ${metrics['MAE']:>12,.0f}")
    print(f"  R²   : {metrics['R2']:.4f}")
    print("--------------------------\n")


def plot_feature_importance(importance_dict, save_path="checkpoints/feature_importance.png"):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], scores[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance chart saved to {save_path}")


def plot_predictions(y_true_log, y_pred_log, save_path="checkpoints/pred_vs_actual.png"):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Predicted vs Actual House Prices")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction scatter plot saved to {save_path}")
