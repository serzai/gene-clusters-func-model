import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from train import load_and_split_data


def evaluate_model(
    data_path: str, model_path: str, output_dir: str = "reports/figures"
) -> None:
    """
    Loads the trained model and test data, then generates evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Related (0)", "Related (1)"],
        yticklabels=["Not Related (0)", "Related (1)"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"Saved Confusion Matrix to {cm_path}")
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC Curve to {roc_path}")
    plt.close()

    # Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR curve (AP = {avg_precision:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    print(f"Saved PR Curve to {pr_path}")
    plt.close()


if __name__ == "__main__":
    DATA_PATH = "data/processed/pairwise_cogs.csv"
    MODEL_PATH = "models/model_pipeline.joblib"

    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: Model file {MODEL_PATH} not found. Please train the model first."
        )
    elif not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
    else:
        evaluate_model(DATA_PATH, MODEL_PATH)
