import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate(name, y_true, y_pred, verbose=True):
    if verbose:
        print(f"=== {name} ===")
        print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion(name, y_true, y_pred, normalize="true", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize=normalize,
        display_labels=range(1, 8),
        values_format=".2f" if normalize else "d",
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix - {name}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()