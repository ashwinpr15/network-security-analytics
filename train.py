import os
import random
import json
import joblib
from pathlib import Path
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
# Split raw records before fitting any transformation.
train_validation, test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
)
train, validation = train_test_split(
    train_validation, test_size=0.25, random_state=RANDOM_SEED,
    stratify=train_validation['label']
)
X_train, y_train, preprocessor = preprocess_data(train)
X_validation, y_validation, _ = preprocess_data(validation, preprocessor)
X_test, y_test, _ = preprocess_data(test, preprocessor)

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
    validation_data=(X_validation, y_validation),
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
Path('models').mkdir(exist_ok=True)
model.save('models/ids_cnn_model.keras')
joblib.dump(preprocessor, 'models/preprocessor.joblib')
report = {
    'dataset': 'generated synthetic connection records; not NSL-KDD',
    'seed': RANDOM_SEED,
    'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test)},
    'roc_auc': float(roc_auc),
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
}
Path('models/evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print("\nModel saved to models/ids_cnn_model.keras")
