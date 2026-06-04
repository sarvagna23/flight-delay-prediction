import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ingest import load_data
from preprocess import preprocess

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'flight_model.pkl')

def train_model(X_train, y_train):
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training complete")
    return model

def evaluate_model(model, X_test, y_test, features):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Results ---")
    print(classification_report(y_test, y_pred, target_names=['On Time', 'Delayed']))

    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC Score:  {roc_auc:.4f}")
    print(f"PR-AUC Score:   {pr_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")

    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nTop 5 Features:\n{importance.head().to_string()}")

    return roc_auc, pr_auc, accuracy

def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, features)
    save_model(model)