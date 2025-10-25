import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb  
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title='Income Prediction with XGBoost + SHAP', layout='wide')

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    
    model.load_model('models/base_model.json')
    
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()


st.title('Income Level Prediction')
st.markdown('Predict whether the income exceeds **$50K/year** based on [Adult Census Income data](https://www.kaggle.com/datasets/uciml/adult-census-income).')
st.markdown('Built with **XGBoost** and explained using **SHAP values**.')


st.sidebar.header('Input Features')

def user_input():
    age = st.sidebar.slider('Age', 0, 100, 50)
    
    workclass = st.sidebar.selectbox('Workclass',
                                     ['Private','State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Self-emp-inc',
                                      'Local-gov','Without-pay','Never-worked'])
    
    education = st.sidebar.selectbox('Education', 
                                     ['HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate', 'Prof-school', 'Bachelors', 'Masters',
                                     '11th', 'Assoc-acdm', 'Assoc-voc','1st-4th', '5th-6th', '12th', '9th', 'Preschool'])
        
    education_num = {
        'Preschool': 1,
        '1st-4th': 2,
        '5th-6th': 3,
        '7th-8th': 4,
        '9th': 5,
        '10th': 6,
        '11th': 7,
        '12th': 8,
        'HS-grad': 9,
        'Some-college': 10,
        'Assoc-voc': 11,
        'Assoc-acdm': 12,
        'Bachelors': 13,
        'Masters': 14,
        'Prof-school': 15,
        'Doctorate': 16
    }
    
    marital_status = st.sidebar.selectbox('Marital Status',
                                          ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-civ-spouse',
                                           'Married-spouse-absent', 'Married-AF-spouse'])
    
    occupation = st.sidebar.selectbox('Occupation',
                                      ['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty', 'Other-service',
                                       'Adm-clerical', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners',
                                       'Sales', 'Farming-fishing', 'Tech-support', 'Protective-serv',
                                       'Armed-Forces', 'Priv-house-serv'])
    
    relationship = st.sidebar.selectbox('Relationship',
                                        ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative', 'Husband', 'Wife'])
    
    race = st.sidebar.selectbox('Race',
                                ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    
    capital_gain = st.sidebar.slider('Capital Gain', 0, 9999, 0)
    
    capital_loss = st.sidebar.slider('Capital Loss', 0, 9999, 0)
    
    hours_per_week = st.sidebar.slider('Hours per week', 1, 100, 40)
    
    native_country = st.sidebar.selectbox('Native Country',
                                          ['United-States', 'Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan', 'India', 'Philippines',
                                           'Trinadad&Tobago', 'Canada', 'South', 'Holand-Netherlands', 'Puerto-Rico',
                                           'Poland', 'Iran', 'England', 'Germany', 'Italy', 'Japan', 'Hong',
                                           'Honduras', 'Cuba', 'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
                                           'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala', 'Jamaica', 'Ecuador',
                                           'France', 'Yugoslavia', 'Scotland', 'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)'])

    data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'education_num': [education_num[education]],
        'marital_status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital_gain': [capital_gain],
        'capital_loss': [capital_loss],
        'hours_per_week': [hours_per_week],
        'native_country': [native_country]
    })
    
    return data

input = user_input()
input = pd.get_dummies(input)

feature_names = model.get_booster().feature_names
input = input.reindex(columns= feature_names, fill_value= 0)

prediction = model.predict(input)[0]
prob = model.predict_proba(input)[0][1]

st.subheader('Prediction')
if prediction == 1:
    st.success(f'Predicted Income: > $50K/year (Probability: {prob:.2f})')
else:
    st.warning(f'Predicted Income: ≤ $50K/year (Probability: {prob:.2f})')


st.subheader('SHAP Explanation')

shap_values = explainer(input)
fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values.base_values[0],
                                     data=input.iloc[0],
                                     feature_names=input.columns))
st.pyplot(fig)
