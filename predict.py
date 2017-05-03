#from make_models2 import main
from data_cleaning import DataCleaning
import cPickle as pickle
from pymongo import MongoClient
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json
import time

def predict(model, data_path):

    original_df = pd.DataFrame.from_dict([data_path])

    print "flag"
    if collection.find({'object_id': {"$eq": original_df['object_id'][0]}}).count() > 0:
        return
    cleaner = DataCleaning(data_path, predict=True)
    cleaned_df = cleaner.clean(predict=True)
    #import ipdb; ipdb.set_trace()
    for i, row in enumerate(cleaned_df.iterrows()):

        row_data = np.array([field for field in row[1]]).reshape(1,-1)
        prob = model.predict_proba(row_data)[:,1][0]
        record = original_df.iloc[i,:].to_dict()
        record['prediction'] = prob
        record['timestamp'] = datetime.utcnow()
        #strftime(format="%Y-%m-%dT%H:%M:%SZ")
        r_copy = record.copy()
        add_to_db(record, collection)
        return r_copy

def add_to_db(row, collection):
    collection.insert_one(row)


def setup_db():
    client = MongoClient()
    db = client['prediction_database']
    preds = db['prediction_collection']
    # result = preds.delete_many({})
    # print "records deleted: ", result.deleted_count
    return preds

def pull_loop(model):

    while True:
        data_path = requests.get("http://galvanize-case-study-on-fraud.herokuapp.com/data_point").json()
        predict(model, data_path)
        time.sleep(1)


if __name__ == '__main__':


    pickle_path = "model_files/model_GradientBoostingClassifier.pkl"
    collection = setup_db()

    #collection.insert_one(data_path)
    with open(pickle_path) as f:
        model = pickle.load(f)
    pull_loop(model)
