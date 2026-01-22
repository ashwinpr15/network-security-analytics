import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data_utils import generate_network_data, preprocess_data, RANDOM_SEED
from src.model import build_cnn_model

# 1. Reproducibility Settings
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 2. Data Pipeline
print("Generating Network Traffic Logs (Synthetic NSL-KDD)...")
df = generate_network_data(n_samples=10000, include_categorical=True)
X, y, preprocessor = preprocess_data(df)

# Stratified Split (Crucial for Imbalanced Data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape:  {X_test.shape}")

# 3. Build Model
model = build_cnn_model(input_shape=(X_train.shape[1], 1))
model.summary()

# 4. Train with Early Stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=2
)

# 5. Evaluation Metrics
print("\n--- PERFORMANCE REPORT ---")
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Save Model
model.save('models/ids_cnn_model.h5')
print("\nModel saved to models/ids_cnn_model.h5")
