import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, dropout=0.3):
    """
    Constructs a 1D-CNN with BatchNormalization and Dropout.
    """
    model = models.Sequential()

    # 1. Feature Extraction
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout))

    # 2. Classification
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary Output

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model
