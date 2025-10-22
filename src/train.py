import xgboost as xgb
from preprocess import train_test_from_path
import os

def train_models(data_path):
    
    X_train, X_test, y_train, y_test = train_test_from_path(data_path)

    print('Training base model...')
    base_model = xgb.XGBClassifier()
    base_model.fit(X_train, y_train)


    base_model.save_model('models/base_model.json')
    print("Saved base model")



    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

    weighted_model = xgb.XGBClassifier(scale_pos_weight= neg_pos_ratio)
    weighted_model.fit(X_train, y_train)


    weighted_model.save_model('models/balanced_model.json')
    print('Saved balanced model')


train_models("data/adult_raw.csv")
