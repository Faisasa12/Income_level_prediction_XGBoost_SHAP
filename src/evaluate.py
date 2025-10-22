import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from preprocess import train_test_from_path

def evaluate_model(model_path, X_test, y_test, model_name="Model"):
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc}")

    false_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_prob)
    plt.plot(false_pos_rate, true_pos_rate, label=f"{model_name} (AUC = {roc_auc})")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    plt.title(f"ROC Curve - {model_name}")
    plt.grid(True)
    plt.show()


X_train, X_test, y_train, y_test = train_test_from_path("data/adult_raw.csv")

evaluate_model('models/base_model.json', X_test, y_test, model_name="Base XGBoost")
evaluate_model('models/balanced_model.json', X_test, y_test, model_name="Balanced XGBoost")
