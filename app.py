import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

#####
# Data import
df = pd.read_csv("bank.csv")

#####
def get_valid_categories(df, column):
    """
    Obtain list of available categories for column
    
    Inputs:
        df (pandas.DataFrame): dataframe from which to extract column values
        column (str): target column for which to extract values
    
    Returns:
        categories: A list of potential values for column
    """
    categories = list(df[column].unique())

    return categories


########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    #obs_dict = request.get_json()
    request_dict = request.get_json()
    base_dict_keys = ['observation_id', 'data']
    data_dict_keys = ['age','workclass','sex','race','education','marital-status', 'capital-gain', 
                      'capital-loss', 'hours-per-week']
    category_columns = ['workclass','sex','race','education','marital-status']
    numeric_colums = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    if list(request_dict.keys())[0] != base_dict_keys[0]:
        return jsonify({
                "observation_id": None,
                "error": "Missing observation_id"
                    })
    if list(request_dict.keys())[1] != base_dict_keys[1]:
        return jsonify({
                "observation_id": None,
                "error": "Missing data"
                    })
    for key in data_dict_keys:
        if key not in list(request_dict['data'].keys()):
            return jsonify({
                "observation_id": request_dict['observation_id'],
                "error": "Missing " + key
                    })
    for key in list(request_dict['data'].keys()):
        if key not in data_dict_keys:
            return jsonify({
                "observation_id": request_dict['observation_id'],
                "error": key + ' not recognized'
                    })
    for cat in category_columns:
        if request_dict['data'][cat] not in get_valid_categories(df, cat):
            return jsonify({
                    "observation_id": request_dict['observation_id'],
                    "error": request_dict['data'][cat] + " incorrect value for " + cat
                    })
    for cat in numeric_colums:
        if (request_dict['data'][cat] > df[cat].max()) or (request_dict['data'][cat] < df[cat].min()):
            return jsonify({
                    "observation_id": request_dict['observation_id'],
                    "error": str(request_dict['data'][cat]) + " incorrect value for " + cat
                    })
    
    #### Prediction
    pred_prob = pipeline.predict_proba(pd.DataFrame([{
            "age": request_dict['data']['age'], 
            "workclass": request_dict['data']['workclass'], 
            "education": request_dict['data']['education'], 
            "marital-status": request_dict['data']['marital-status'], 
            "race": request_dict['data']['race'],
            "sex": request_dict['data']['sex'],
            "capital-gain": request_dict['data']['capital-gain'], 
            "capital-loss": request_dict['data']['capital-loss'], 
            "hours-per-week": request_dict['data']['hours-per-week']}
        ], columns=columns).astype(dtypes))[0][1]
    
    if pred_prob > 0.5:
        pred = True
    else:
        pred = False
    response = {
                "observation_id": request_dict['observation_id'],
                "prediction": pred,
                "probability": pred_prob
            }
    
    return jsonify(response)


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
