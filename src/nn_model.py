import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_nn_model(input_dim=10, num_classes=3,
                    layer_0_units=64, layer_1_units=32, layer_2_units=16,
                    layer_0_dropout=0.3, layer_1_dropout=0.3, layer_2_dropout=0.3,
                    optimizer='adam'):
    """
    Creates a simple neural network model using Keras.

    Args:
        input_dim (int): The number of input features.
        num_classes (int): The number of output classes for classification.
        layer_0_units (int): Number of units in the first dense layer.
        layer_1_units (int): Number of units in the second dense layer.
        layer_2_units (int): Number of units in the third dense layer.
        layer_0_dropout (float): Dropout rate for the first hidden layer. # Clarified
        layer_1_dropout (float): Dropout rate for the second hidden layer.
        layer_2_dropout (float): Dropout rate for the third hidden layer.
        optimizer (str): Optimizer to use for compilation.

    Returns:
        keras.Model: A Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(layer_0_units, activation='relu'),
        layers.Dropout(layer_0_dropout), # Applied layer_0_dropout to first HIDDEN layer
        layers.Dense(layer_1_units, activation='relu'),
        layers.Dropout(layer_1_dropout), # Applied layer_1_dropout to second HIDDEN layer
        layers.Dense(layer_2_units, activation='relu'),
        layers.Dropout(layer_2_dropout), # Applied layer_2_dropout to third HIDDEN layer
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Example usage:
    model = create_nn_model(input_dim=20, num_classes=5)
    model.summary()