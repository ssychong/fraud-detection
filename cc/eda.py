import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from smote_cc import smote

df = pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created','event_end','event_published','event_start','user_created','event_published'])


df['fraud'] = df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])
Y = df.pop('fraud').values
X = df.values
cols = df.columns
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.8, random_state=123)
training = pd.DataFrame(X_train, columns=cols)
training['fraud'] = y_train
test = pd.DataFrame(X_test, columns=cols)
test['fraud'] = y_test



X_smote, y_smote = smote(X_test, y_test, 0.3)
