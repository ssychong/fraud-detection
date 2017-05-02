import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from smote_cc import smote, oversaple
import numpy as np
from data_cleaning import DataCleaning

#df = pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created','event_end','event_published','event_start','user_created','event_published'])


# df['fraud'] = df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])
# Y = df.pop('fraud').values
# X = df.values
# cols = df.columns
# X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.8, random_state=123)
# training = pd.DataFrame(X_train, columns=cols)
# training['fraud'] = y_train
# test = pd.DataFrame(X_test, columns=cols)
# test['fraud'] = y_test



# bin_func = np.vectorize(binary_mask)
# y_train_bin = bin_func(y_train)
# X_train_notext = X_train.ix[:, X_train.columns != 'description']

dc = DataCleaning('data/data.json', training=True)
X_clean, y_clean = dc.clean()

X_over, y_over = oversample(X_clean, y_clean, 0.3)
