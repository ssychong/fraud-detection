from sklearn.ensemble import GradientBoostingClassifier
import cPickle as pickle

gb = GradientBoostingClassifier()

with open('gb_model.pkl', 'w') as f:
        pickle.dump(gb, f)
