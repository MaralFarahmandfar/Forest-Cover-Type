import re

import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.data import PROJECT_ROOT
from src.models import build_model
from src.train import train_model, plot_history, predict_classes
from src.evaluate import evaluate, plot_confusion

RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(name, hidden_units, data, results, seed=42,
                   learning_rate=1e-3, batch_size=512, epochs=20, **build_kwargs):
    """Run one full experiment: build, train, plot, evaluate on dev, log.

    Args:
        name: experiment name, used in titles and file names.
        hidden_units: list with the neurons of each hidden layer.
        data: dict from load_processed() with the six arrays.
        results: list where the metrics dict of this experiment is appended.
        seed: random seed, set at the start so every run is comparable.
        learning_rate, batch_size, epochs: training settings.
        **build_kwargs: extra arguments passed to build_model
            (for example l2 or dropout, once they are implemented).

    Returns:
        The trained model.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # "NN 2x(128,64)" -> "nn_2x_128_64"
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    tf.keras.utils.set_random_seed(seed)
    model = build_model(hidden_units, **build_kwargs)
    history = train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_dev"], data["y_dev"],
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

    plot_history(history, title=name, save_path=RESULTS_DIR / f"curves_{slug}.png")

    y_pred = predict_classes(model, data["X_dev"])
    metrics = evaluate(name, data["y_dev"], y_pred)

    # Accuracy on the full training set after training (for bias/variance analysis)
    train_pred = predict_classes(model, data["X_train"])
    metrics["params"] = model.count_params()
    metrics["train_acc"] = accuracy_score(data["y_train"], train_pred)

    plot_confusion(name, data["y_dev"], y_pred, save_path=RESULTS_DIR / f"cm_{slug}.png")

    results.append(metrics)
    return model