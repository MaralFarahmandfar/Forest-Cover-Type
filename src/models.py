import tensorflow as tf


def build_model(hidden_units, initializer="he_normal", n_features=54, n_classes=7):
    """Build a fully connected network for Covertype (not compiled).

    Args:
        hidden_units: list with the number of neurons of each hidden layer,
            e.g. [64] is one hidden layer, [128, 64] is two.
        initializer: kernel initializer of the hidden layers
            ("he_normal" is the He initialization, suited for ReLU).
        n_features: number of input features.
        n_classes: number of output classes (softmax).

    Returns:
        A tf.keras.Sequential model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(n_features,)))

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_initializer=initializer))

    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    return model