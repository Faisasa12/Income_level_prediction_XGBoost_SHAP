import pandas as pd
import numpy as np
import xgboost as xgb  
import argparse
from preprocess import processed_data_from_input


parser = argparse.ArgumentParser(description="Predict income using trained model")
parser.add_argument('--input', type=str, required=True, help="CSV file path")
parser.add_argument('--model', type=str, required=True, help="Saved model path")

args = parser.parse_args()


data = pd.read_csv(args.input)

input = processed_data_from_input(data)

model = xgb.XGBClassifier()
model.load_model(args.model)
feature_names = model.get_booster().feature_names

input = input.reindex(columns= feature_names, fill_value= 0)

preds = model.predict(input)
probs = model.predict_proba(input)[:, 1]

results = data.copy()
results['prediction'] = preds
results['probability'] = probs

print(results)
