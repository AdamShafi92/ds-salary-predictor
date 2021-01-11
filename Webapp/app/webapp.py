import streamlit as st
import requests
import datetime
import shap
import json
import pickle
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from helper import *

clf = xgb.Booster()  # init model
clf.load_model('xgb_model.bst')  # load data

with open('list_options.pkl', 'rb') as pkl_item:
    locations, employees, industries = pickle.load(pkl_item)


st.title('UK Data Jobs Salary Predictor')

# description and instructions
st.write("""
- This tool predicts salaries for data-related jobs in the UK using a Machine Learning model (XGBoost). \n
- The model has been trained on data from Indeed.co.uk, scraped in Q3-Q4 2020. \n
- The model has been tested and has a Mean Absolute Error score of around £9000. \n
- It also provides explainer scores using SHAP. These show which parts of the job spec are influencing the prediction. \n
- Please note, if a word wasn't in the training dataset, the model won't use it to make predictions. \n
""")

st.sidebar.header('Job Description Data')

def user_input_features():
    input_features = {}
    input_features["Title"] = st.sidebar.text_input('Job Title')
    input_features["Location"] = st.sidebar.selectbox('Location (Select Closest)', locations)
    input_features["Employees"] = st.sidebar.selectbox('Company Size', employees)
    input_features["Industry"] = st.sidebar.selectbox('Industry', industries)
    input_features["description"] = st.sidebar.text_input('Job Description')
    return [input_features]

json_data = user_input_features()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# read pickle files
with open('score_objects.pkl', 'rb') as handle:
    d, features_selected, explainer = pickle.load(handle)

# explain model prediction results
def explain_model_prediction(json_data):
    X = pd.DataFrame.from_dict(json_data)
    X = d.transform(X)
    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], X)
    return p, shap_values

def predict(json_data):

    data = pd.DataFrame(json_data)
    
    #transform DataFrame
    data['Title'] = clean_text(data['Title'])
    data['description'] = clean_text(data['description'])
    
    #score df
    prediction = predict_pipe(data,d,clf)
      
    return prediction

def predict_shap(json_data):
    X = pd.DataFrame.from_dict(json_data)
    X = d.transform(X)  
    shap_values = explainer.shap_values(X)
    feats = pd.DataFrame(X.toarray(),columns = features_selected)
    
    return shap_values, feats


submit = st.sidebar.button('Get predictions')
if submit:
    
    prediction = predict(json_data)
    shap_values,feats = predict_shap(json_data)
    
    st.header('Results')
    st.write(f"Predicted Salary:  £{round(float(prediction),2)}")
    
    #explainer force_plot
    
    
    st.subheader('Force Plot')
    st.write('''This shows how the feature influences the predictions. 
             If the value = 0, the lack of the feature is the influence.''')
    p = shap.force_plot(explainer.expected_value, shap_values, feats)
    st_shap(p)
    
    st.subheader('Decision Plot')
    st.write('''This shows how the model reached is prediction, showing each feature more clearly.''')    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.decision_plot(explainer.expected_value, shap_values, feats)
    st.pyplot(fig)
  

    st.subheader('Summary Plot 1')
    st.write('''This shows the £ value of each feature used in the prediction.''')    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, feats)
    st.pyplot(fig)
