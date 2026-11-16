import tensorflow as tf


def build_model(hidden_units, initializer="he_normal", n_features=54, n_classes=7, l2=0.0, dropout=0.0):
    """Build a fully connected network for Covertype (not compiled).

    Args:
        hidden_units: list with the number of neurons of each hidden layer,
            e.g. [64] is one hidden layer, [128, 64] is two.
        initializer: kernel initializer of the hidden layers
            ("he_normal" is the He initialization, suited for ReLU).
        n_features: number of input features.
        n_classes: number of output classes (softmax).
        l2: L2 regularization strength for the hidden layers (0 = off).
        dropout: fraction of neurons dropped after each hidden layer
            (0 = off). Note: this is the drop rate, not keep_prob.

    Returns:
        A tf.keras.Sequential model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(n_features,)))

    regularizer = tf.keras.regularizers.l2(l2) if l2 > 0 else None

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation="relu",
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizer))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    return model