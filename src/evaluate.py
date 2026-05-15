from train import load_and_split_data
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
import joblib
import argparse
import warnings
import os

os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")


def evaluate_model(
    data_path: str, model_path: str, output_dir: str = "reports/figures"
) -> None:
    """
    Loads the trained model and test data, then generates evaluation plots.
    """
    import warnings

    warnings.filterwarnings("ignore")

    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)

    _, X_test, _, y_test = load_and_split_data(data_path)
    print(f"Evaluating on {len(X_test)} test samples...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2e",
        cmap="Blues",
        xticklabels=["Not Related", "Related"],
        yticklabels=["Not Related", "Related"],
        cbar_kws={"label": "Count"},
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    sns.despine()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"Saved Confusion Matrix plot to {cm_path}")
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="#0072B2",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.2, linestyle="--")
    sns.despine()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC Curve plot to {roc_path}")
    plt.close()

    # Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(
        recall,
        precision,
        color="#D55E00",
        lw=2,
        label=f"PR curve (AUC = {avg_precision:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.2, linestyle="--")
    sns.despine()
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    print(f"Saved PR Curve plot to {pr_path}")
    plt.close()

    # Feature Importance Analysis
    plot_feature_importance(pipeline, X_test, y_test, output_dir)


def plot_feature_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
    n_repeats: int = 10,
) -> None:
    """
    Calculates and plots Permutation Importance for the model.
    """
    import warnings

    warnings.filterwarnings("ignore")

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()
    importance_df = pd.DataFrame(
        result.importances[sorted_idx].T, columns=X_test.columns[sorted_idx]
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=importance_df, orient="h", color="#56B4E9", fliersize=1)
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    plt.xlabel("Decrease in model accuracy")
    plt.title("Feature Importance")
    plt.tight_layout()
    sns.despine()

    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    print(f"Saved Feature Importance plot to {fi_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/pairwise_cogs.csv",
        help="Path to the processed *.csv file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model_pipeline.joblib",
        help="Path to the trained model pipeline.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports/figures",
        help="Directory to save evaluation plots.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(
            f"Error: Model file {args.model} not found. Please train the model first."
        )
    elif not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
    else:
        evaluate_model(args.data, args.model, args.out_dir)
