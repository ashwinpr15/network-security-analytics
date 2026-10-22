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

# Baseline for context: is the CNN better than a plain logistic regression on the same features?
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
baseline = LogisticRegression(max_iter=2000).fit(X_train.reshape(len(X_train), -1), y_train)
baseline_proba = baseline.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
baseline_scores = {
    'accuracy': float(accuracy_score(y_test, (baseline_proba >= 0.5).astype(int))),
    'roc_auc': float(roc_auc_score(y_test, baseline_proba)),
}
print(f"Logistic regression baseline: accuracy {baseline_scores['accuracy']:.4f}, ROC AUC {baseline_scores['roc_auc']:.4f}")

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
    'epochs_run': len(history.history['loss']),
    'logistic_regression_baseline': baseline_scores,
}
Path('models/evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print("\nModel saved to models/ids_cnn_model.keras")

# Charts for the README: training curves and the test confusion matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Path('results').mkdir(exist_ok=True)
Path('results/metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
epochs = range(1, len(history.history['loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs, history.history['loss'], marker='o', label='Train')
axes[0].plot(epochs, history.history['val_loss'], marker='o', label='Validation')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].plot(epochs, history.history['accuracy'], marker='o', label='Train')
axes[1].plot(epochs, history.history['val_accuracy'], marker='o', label='Validation')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
fig.tight_layout()
fig.savefig('results/training_curves.png', dpi=120)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks([0, 1], ['Normal', 'Attack']); ax.set_yticks([0, 1], ['Normal', 'Attack'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Test set (ROC AUC {roc_auc:.3f})')
fig.tight_layout()
fig.savefig('results/confusion_matrix.png', dpi=120)
print("Saved results/metrics.json, results/training_curves.png and results/confusion_matrix.png")
