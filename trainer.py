# trainer.py

import os
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ..config import SETTINGS

def train_model():
    """Loads data, trains, evaluates, and saves the LightGBM model."""
    data_path = SETTINGS.get_labeled_data_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Labeled data file not found at '{data_path}'. Run labeling first.")

    print(f"--- Starting Model Training ---")
    df = pd.read_parquet(data_path)
    cfg = SETTINGS.training
    df.dropna(subset=cfg.FEATURE_COLUMNS + [cfg.TARGET_COLUMN], inplace=True)
    print(f"Loaded {len(df):,} clean rows for training.")

    X = df[cfg.FEATURE_COLUMNS].copy()
    y = df[cfg.TARGET_COLUMN] + 1  # Remap labels from [-1, 0, 1] to [0, 1, 2]
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    print(f"Splitting data chronologically: {1-cfg.TEST_SET_SIZE:.0%} train, {cfg.TEST_SET_SIZE:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SET_SIZE, shuffle=False
    )
    
    model = lgb.LGBMClassifier(**cfg.LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    print("Model training complete.")

    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    target_names = ["Loss", "Timeout", "Profit"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set"); plt.show()
    
    lgb.plot_importance(model, max_num_features=20, importance_type='gain')
    plt.title("Feature Importance (Gain)"); plt.tight_layout(); plt.show()
    
    model_path = SETTINGS.get_model_path()
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved successfully to: {model_path}")
