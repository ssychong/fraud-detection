#from make_models2 import main
from data_cleaning import DataCleaning
import cPickle as pickle
from pymongo import MongoClient
import numpy as np
import pandas as pd

def unpickle_predict(data_path, pickle_path, collection):
    with open(pickle_path) as f:
        model = pickle.load(f)
    original_df = pd.read_json(data_path)
    cleaner = DataCleaning(data_path, predict=True)
    cleaned_df = cleaner.clean(predict=True)
    for i, row in enumerate(cleaned_df.iterrows()):
        row_data = np.array([field for field in row[1]]).reshape(1,-1)
        prob = model.predict_proba(row_data)[:,1][0]
        record = original_df.iloc[i,:].to_dict()
        record['prediction'] = prob
        add_to_db(record, collection)

def add_to_db(row, collection):
    collection.insert_one(row)


def setup_db():
    client = MongoClient()
    db = client['prediction_database']
    preds = db['prediction_collection']
    result = preds.delete_many({})
    print "records deleted: ", result.deleted_count
    return preds


if __name__ == '__main__':
    data_path = "data/subset.json"
    pickle_path = "model_RandomForestClassifier.pkl"
    collection = setup_db()
    unpickle_predict(data_path, pickle_path, collection)
