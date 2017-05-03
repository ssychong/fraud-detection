from flask import Flask, render_template, jsonify
from predict import predict, add_to_db, setup_db
import cPickle as pickle
import requests
from pymongo import MongoClient
import pymongo
import json
from time import sleep
app = Flask(__name__)

client = MongoClient()
db = client['prediction_database']
collection = db['prediction_collection']

@app.route('/')
def index():
    text = """Welcome to the page."""
    return render_template("index.html", text=text)

@app.route('/score', methods=['GET'])
def score():
    data_path = (requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').json())
    record = predict(model, data_path, collection)
    print record
    return jsonify(record)

@app.route('/dashboard')
def dashboard():
    results = collection.find().sort("timestamp", pymongo.DESCENDING).limit(1)[0]
    pred = results['prediction']
    return render_template("index.html", prediction=pred, text=results)
    return str(results[0])


if __name__ == '__main__':
    data_path = "data/subset.json"
    pickle_path = "model_files/model_GradientBoostingClassifier.pkl"
    with open(pickle_path) as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8000, debug=True)
