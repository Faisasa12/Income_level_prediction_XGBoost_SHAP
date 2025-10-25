import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def train_test_from_path(csv_path):
    dataset = pd.read_csv(csv_path)
    
    dataset.columns = [col.replace(".",'_') for col in dataset.columns]
    dataset = dataset.drop('fnlwgt', axis=1)
    dataset.replace('?', np.nan, inplace=True)
    
    dataset.dropna(inplace=True)
    
    features = dataset.drop('income', axis=1)
    target = dataset['income']
    
    features = pd.get_dummies(features)
    target = target.map({
        '<=50K': 0,
        '>50K': 1
    })
    
    return train_test_split(features, target, test_size = 0.2, stratify = target)

def train_test_from_raw_data(dataset):
    dataset.columns = [col.replace(".",'_') for col in dataset.columns]
    dataset = dataset.drop('fnlwgt', axis=1)
    dataset.replace('?', np.nan, inplace=True)
    
    dataset.dropna(inplace=True)
    
    features = dataset.drop('income', axis=1)
    target = dataset['income']
    
    features = pd.get_dummies(features)
    target = target.map({
        '<=50K': 0,
        '>50K': 1
    })
    
    return train_test_split(features, target, test_size = 0.2, stratify = target)

def processed_data_from_input(dataset):
    og_dataset = pd.read_csv('data/adult_raw.csv')
    x_train, x_test, y_train, y_test = train_test_from_raw_data(og_dataset)
    
    dataset.columns = [col.replace(".",'_') for col in dataset.columns]
    
    if 'fnlwgt' in dataset:
        dataset = dataset.drop('fnlwgt', axis=1)
        
    dataset.replace('?', np.nan, inplace=True)
    
    dataset.dropna(inplace=True)
    
    dataset = pd.get_dummies(dataset)
    
    dataset = dataset.reindex(columns=x_train.columns, fill_value=0)
    print(dataset)
    
    return dataset