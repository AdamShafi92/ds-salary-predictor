import flask
import pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector

from custom_funcs import *



app = flask.Flask(__name__, template_folder='templates')

with open('./model/col_trans.pkl', 'rb') as f:
    col_trans = pickle.load(f)
with open('./model/pipe.pkl', 'rb') as f:
    pipe = pickle.load(f) 


@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    
    if flask.request.method == 'POST':
        url = flask.request.form['url']
        df = get_job(url)
        title_ = df['Title'].iloc[0]
        company_ = df['Company'].iloc[0]
        location_ = df['Location'].iloc[0]
        df = get_company(df)
        df_clean=(df.pipe(clean_location)
                    .pipe(clean_text,col='Title')
                    .pipe(clean_text,col='description')
                    .pipe(clean_df))
        df_prep = col_trans.transform(df_clean)
        prediction = pipe.predict(df_prep)
        if prediction == 1:
            prediction='Yes'
        else:
            prediction='No'

        return flask.render_template('main.html',
                                     original_input={'URL':url,
                                                     'TITLE':title_,
                                                     'COMPANY':company_,
                                                     'LOCATION':location_
                                                    },
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()


