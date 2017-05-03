from flask import Flask, render_template, jsonify
from predict import predict, add_to_db, setup_db
import cPickle as pickle
import requests
import pymongo
import json
from time import sleep

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Fyre!')

@app.route('/more/')
def more():
    return render_template('prediction_template.html')

@app.route('/data_cleaning')
def data_cleaning():
    return render_template('dc_template.html')

@app.route('/features')
def features():
    return render_template('feat_template.html')

@app.route('/modeling')
def modeling():
 return render_template('model_template.html')


@app.route('/score', methods=['GET'])
def score():
    data_path = (requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').json())
    record = predict(model, data_path)
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
    collection = setup_db()
    with open(pickle_path) as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
