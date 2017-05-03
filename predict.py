from make_models2 import main
from data_cleaning import DataCleaning
import cPickle as pickle

subset = 'data/subset.json'

predictions = []

# def unpickle_predict(path):
dc = DataCleaning(subset)
X_train, y_train = dc.clean()
X = X_train.reshape(1, -1)
for idx, line in enumerate(X):
    with open('gb_model.pkl') as f:
        gb_model = pickle.load(f)
    gb_model.fit(line, y_train[idx])
    pred = gb_model.predict(line)
    prob = gb_model.predict_proba(line)
    print(pred, prob)
