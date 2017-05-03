#from make_models2 import main
from data_cleaning import DataCleaning
import cPickle as pickle

def unpickle_predict(data_path, pickle_path):
    predictions = []
    with open(pickle_path) as f:
        model = pickle.load(f)
    cleaned_example = DataCleaning(data_path)
    X_train, y_train = cleaned_example.clean(predict=True)
    for row in X_train:
        row_data = row.reshape(1, -1)
        prob = model.predict_proba(row_data)[:,1][0]
        predictions.append(prob)
    return predictions

if __name__ == '__main__':
    data_path = "data/subset.json"
    pickle_path = "model.pkl"
    saved = unpickle_predict(data_path, pickle_path)
    print len(saved)
    print saved
