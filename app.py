import os
import json
import pickle
import joblib
import pandas as pd
from uuid import uuid4
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Read dataset
df = pd.read_csv("bank.csv")
########################################
# Import functions needed
def get_valid_categories(df, column):
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
    obs_dict = request.get_json()
    #response = obs_dict
    #_id = obs_dict['id']
    #observation = obs_dict['observation']
    #try:
    #    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    #    proba = pipeline.predict_proba(obs)[0, 1]
    #    response = {'proba': proba}
    #    p = Prediction(
    #        observation_id=_id,
    #        proba=proba,
    #        observation=observation
    #    )
    #    p.save()
    #except IntegrityError:
    #    error_msg = 'Observation ID: "{}" already exists'.format(_id)
    #    response['error'] = error_msg
    #    print(error_msg)
    #    DB.rollback()
    #except ValueError:
    #    error_msg = 'Observation is invalid!'
    #    response = {'error': error_msg}
    #    print(error_msg)
    
    #### Validations
    base_dict_keys = ['observation_id', 'data']
    data_dict_keys = ['age','workclass','sex','race','education','marital-status', 'capital-gain', 
                      'capital-loss', 'hours-per-week']
    category_columns = ['workclass','sex','race','education','marital-status']
    numeric_colums = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    if list(obs_dict.keys())[0] != base_dict_keys[0]:
        response = {
                "observation_id": None,
                "error": "Missing observation_id"
                    }
        return jsonify(response) 
    if list(obs_dict.keys())[1] != base_dict_keys[1]:
        response = {
                "observation_id": None,
                "error": "Missing data"
                    }
        return jsonify(response)
    for key in data_dict_keys:
        if key not in list(obs_dict['data'].keys()):
            response = {
                "observation_id": obs_dict['observation_id'],
                "error": "Missing " + key
                    }
            return jsonify(response)
    for key in list(obs_dict['data'].keys()):
        if key not in data_dict_keys:
            response = {
                "observation_id": obs_dict['observation_id'],
                "error": key + ' not recognized'
                    }
            return jsonify(response)
    for cat in category_columns:
        if obs_dict['data'][cat] not in get_valid_categories(df, cat):
            response = {
                    "observation_id": obs_dict['observation_id'],
                    "error": obs_dict['data'][cat] + " incorrect value for " + cat
                    }
            return jsonify(response)
    for cat in numeric_colums:
        if (obs_dict['data'][cat] > df[cat].max()) or (obs_dict['data'][cat] < df[cat].min()):
            response = {
                    "observation_id": obs_dict['observation_id'],
                    "error": str(obs_dict['data'][cat]) + " incorrect value for " + cat
                    }
            return jsonify(response)
    #### End of validations
    #### Prediction
    pred_prob = pipeline.predict_proba(pd.DataFrame([{
            "age": obs_dict['data']['age'], 
            "workclass": obs_dict['data']['workclass'], 
            "education": obs_dict['data']['education'], 
            "marital-status": obs_dict['data']['marital-status'], 
            "race": obs_dict['data']['race'],
            "sex": obs_dict['data']['sex'],
            "capital-gain": obs_dict['data']['capital-gain'], 
            "capital-loss": obs_dict['data']['capital-loss'], 
            "hours-per-week": obs_dict['data']['hours-per-week']}
        ], columns=columns).astype(dtypes))[0][1]
    
    if pred_prob > 0.5:
        pred = True
    else:
        pred = False
    response = {
                "observation_id": obs_dict['observation_id'],
                "prediction": pred,
                "probability": pred_prob
            }

    return jsonify(response)


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
