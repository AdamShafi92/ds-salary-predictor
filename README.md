# Data Jobs Salary Predictor

Visit the Webapp here: https://data-salary-predictor.herokuapp.com/

### Data Collection & Modelling

The aim of this project is to predict salaries of Data-related jobs (eg Data Scientist, Data Analyst, Data Engineer).

This could be used by applicants looking to understand what the market rate is, or by companies to check where their pay is relative the market. It can also help people understand what skills lead to higher pay.

The model was created using data scraped from Indeed.co.uk. This was cleaned using Pandas.

Modelling was done using XGBoost and resulted in a MAE of £8800.

The key limitations of this model are around the data source. Indeed is widely used by large recruitment companies meaning the training data is biased towards the salaries these companies advertise - it may not reflect what people are actually being paid.

The final dataset was relatively small (around 2500 records) which means there may not be enough data to achieve stastical significance for some words.

### Webapp

This was created using Streamlit for a clean frontend. The app also produces SHAP value charts, which can be used to help with interpretation. This are all created within streamlit when the Get Predictions button is pressed.

A Docker version was also created, which separates the frontend Streamlit app and a backend Flask app making predictions. This has a docker-compose file and could be forked and recreated anywhere.
