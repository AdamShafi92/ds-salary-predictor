import shap
import pickle

import spacy
from spacy.lang.en import English
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

import xgboost as xgb



def clean_text(array):
    stopwords = English.Defaults.stop_words   
    array=(array
            .apply(lambda x: ' '.join(x.split('\\n')))
            .str.lower()
            .str.replace('[0-9£–]','',regex=True)
            .apply(word_tokenize)
            .apply(lambda x: [i for i in x if i not in stopwords])
            .apply(pos_tag)
            .apply(lemmatize_sentence)
            .apply(' '.join)
            .apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))
            .str.strip()) 
    return array

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in tokens:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def col_trans(data,trans):
    X = trans.transform(data)
    return xgb.DMatrix(X)

def predict_pipe(data,trans,model):
    X = col_trans(data, trans)
    return model.predict(X)










