import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def make_optimizer(name, learning_rate, momentum=0.9):
    """Return a Keras optimizer from its name.

    Args:
        name: "sgd", "momentum", "rmsprop" or "adam".
        learning_rate: learning rate of the optimizer.
        momentum: momentum coefficient (only used by "momentum").
    """
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if name == "momentum":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def train_model(model, X_train, y_train, X_dev, y_dev,
                learning_rate=1e-3, batch_size=512, epochs=20, verbose=2,
                optimizer="adam"):
    """Compile and train a model, validating on the dev set after each epoch.

    Args:
        model: a Keras model (from build_model).
        X_train, y_train: training data (labels are integers 0 to 6).
        X_dev, y_dev: dev data used for validation.
        learning_rate: learning rate of the optimizer.
        batch_size: mini-batch size.
        epochs: number of passes over the training set.
        verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        optimizer: "sgd", "momentum", "rmsprop" or "adam" (see make_optimizer).

    Returns:
        The Keras History object.
    """
    model.compile(
        optimizer=make_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )
    return history


def plot_history(history, title="", save_path=None):
    """Plot train/dev loss and accuracy per epoch side by side."""
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="dev")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="dev")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_compare(histories, key="val_loss", title="", names=None, save_path=None):
    """Plot one metric of several experiments on the same axes.

    Args:
        histories: dict {experiment name: history.history}.
        key: metric to plot, e.g. "val_loss" or "val_accuracy".
        title: figure title.
        names: optional list of experiment names to plot (default: all).
        save_path: optional path to save the figure.
    """
    names = names if names is not None else list(histories)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        ax.plot(histories[name][key], label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(key)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def predict_classes(model, X, batch_size=2048):
    """Return the predicted class (0 to 6) for each row of X."""
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1)