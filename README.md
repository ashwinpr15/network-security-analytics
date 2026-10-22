<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner-light.svg" alt="Synthetic network classification demonstration" width="100%">
</picture>

# Synthetic network classification demonstration

A TensorFlow/Keras Conv1D model classifies generated connection records. The generator creates numeric features, optional categorical protocol/service/flag fields, and rule-based binary labels. It does not download or load NSL-KDD.

## Implementation and evaluation

- `src/data_utils.py`: synthetic generator plus numeric MinMax scaling and categorical one-hot encoding.
- `src/model.py`: Conv1D, batch normalization, pooling, dropout, and dense binary classification.
- `train.py`: stratified 60/20/20 train/validation/test split of raw records. Preprocessing fits on training data only; early stopping uses validation data; final metrics use the held-out test set.

## Run from the repository root

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.txt
python train.py
```

Generated artifacts: `models/ids_cnn_model.keras`, `models/preprocessor.joblib`, and `models/evaluation.json`. Artifacts are ignored by Git; publish a reviewed evaluation record separately when reporting results. Load only trusted model and joblib files.

## Limitations

No real traffic or deployment integration is included. Synthetic labels are generated from the same input features, so performance demonstrates learning the generator's rule. A convolution over tabular feature order is a modeling choice, not proof of temporal traffic analysis. Compare simpler tabular baselines before selecting this architecture for real data.

Preprocessing is fitted on training data only and the test set is never used for early stopping, so test metrics are free of that leakage.

## Targeted checks

Run `python test_evaluation.py` from the repository root after installing dependencies.

## License

Released under the [MIT License](LICENSE).
