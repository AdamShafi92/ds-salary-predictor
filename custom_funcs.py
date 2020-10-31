import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
from nltk import PorterStemmer
import string


def clean_location(df):
    X = df['Location']
    X=X.str.replace('[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}','',regex=True)
    X=X.str.replace(r'[A-Z]{1,2}\d[A-Z\d]?','',regex=True)
    X=X.apply(lambda x: x.strip())
    X=X.str.replace(r'[^\w\s]','',regex=True)
    X=X.str.replace(' ','_')
    df['Location'] = X
    return df
    
def clean_text(df,col):
    X = df[col]
    stemmer = PorterStemmer()
    X=X.str.replace('\nnew','')
    X=X.str.replace('\n','')
    X=X.apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))
    X=X.str.replace('[0-9£–]','',regex=True)
    X=X.apply(lambda x:' '.join([stemmer.stem(i) for i in x.split(' ')])) 
    df[col] = X
    return df

def clean_df(df):
    df.fillna('No Data',inplace=True)
    df['Company'].str.replace('\n','')
    df = df[['Title', 'Company', 'Location', 'Employees', 'Industry', 'Revenue','description']]
    return df

def get_job(url):    
    r = requests.get(url).text
    soup = BeautifulSoup(r,'html.parser')
    data_dict = {}
    try:
        data_dict['Title'] = soup.find('h1', attrs={'class':'icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title'}).text
    except:
        data_dict['Title'] = np.nan

    try:
        data_dict['Company'] = soup.find('div', attrs={'class':'jobsearch-CompanyInfoWithoutHeaderImage'}).text.split('-')[0]
    except:
        data_dict['Company'] = np.nan

    try:
        data_dict['Location'] = soup.find('div', attrs={'class':'jobsearch-CompanyInfoWithoutHeaderImage'}).text.split('-')[1]
    except:
        data_dict['Location'] = np.nan

    try:
        data_dict['description'] = soup.find('div', attrs={'id':'jobDescriptionText'}).text
    except:
        data_dict['description'] = np.nan
    df= pd.DataFrame(data_dict, index=[0])
    df.fillna('No Data',inplace=True)
    return df

def get_company(df):   
    z,x = [],[]
    url = 'https://www.indeed.co.uk/cmp/'+df['Company'].values[0].replace(' ','+')
    r = requests.get(url).text
    soup = BeautifulSoup(r, 'html.parser')
    try:
        z.append([i.text for i in soup.find_all('div', attrs={'class':'cmp-AboutMetadata-itemTitle'})])
    except:
        z.append(np.nan)
    try:
        x.append([i.text for i in soup.find_all('div', attrs={'class':'cmp-AboutMetadata-itemCotent'})])
    except:
        x.append(np.nan)
    company_info = pd.DataFrame(dict(zip(z[0],x[0])),index=[0]) 
    if 'Employees' in company_info.columns:
        df['Employees'] = company_info['Employees']
    else:
        df['Employees'] = 'No Data'

    if 'Industry' in company_info.columns:
        df['Industry'] = company_info['Industry']
    else:
        df['Industry'] = 'No Data'

    if 'Revenue' in company_info.columns:
        df['Revenue'] = company_info['Revenue']
    else:
        df['Revenue'] = 'No Data'
    return df