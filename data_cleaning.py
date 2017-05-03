import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

class DataCleaning(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, filepath, training=True):
        """Reads in data

        Args:
            fileapth (str): location of file with csv data
        """
        self.df = pd.read_json(filepath)
        self.df['fraud'] = self.df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])
        index_list = range(len(self.df))
        X_train, X_test = train_test_split(index_list, train_size=.8, random_state=123)
        training_data = self.df.iloc[X_train,:]
        test_data = self.df.iloc[X_test,:]

        if training:
            self.df = training_data
        else:
            print "using test data"
            self.df = test_data


    def make_target_variable(self):
        """Create the churn column, which is our target variable y that we are trying to predict
        A customer is considered churned if they haven't taken trip in last 30 days"""
        self.df['last_trip_date'] = pd.to_datetime(self.df['last_trip_date'])
        self.df['Churn'] = (self.df.last_trip_date < '2014-06-01').astype(int)

    def dummify(self, columns):
        """Create dummy columns for categorical variables"""
        dummies = pd.get_dummies(self.df[columns], prefix=columns)
        self.df = self.df.drop(columns, axis=1)
        self.df = pd.concat([self.df,dummies], axis=1)

    def drop_date_columns(self):
        """Remove date columns from feature matrix, avoid leakage"""
        self.df.drop('last_trip_date', axis=1, inplace=True)
        self.df.drop('signup_date', axis=1, inplace=True)

    def get_column_names(self):
        """Get the names of columns currently in the dataframe"""
        return list(self.df.columns.values)

    def cut_outliers(self, col):
        """Remove numerical data more than 3x outside of a column's std"""
        std = self.df[col].std()
        t_min = self.df[col].mean() - 3*std
        t_max = self.df[col].mean() + 3*std
        self.df = self.df[(self.df[col] >= t_min) & (self.df[col] <= t_max)]

    def drop_na(self):
        """Generic method to drop all rows with NA's in any column"""
        self.df = self.df.dropna(axis=0, how='any')

    def make_log_no_trips(self):
        """Transform the number of trips column to log scale"""
        self.df['log_trips'] = self.df[(self.df['trips_in_first_30_days'] != 0)].trips_in_first_30_days.apply(np.log)
        self.df['log_trips'] = self.df['log_trips'].apply(lambda x: 0 if np.isnan(x) else x)

    def drop_columns_for_regression(self):
        """Drop one of the dummy columns for when using a regression model"""
        self.df = self.df.drop(['phone_iPhone', 'city_Astapor'], axis=1)

    def mark_missing(self, cols):
        """Fills in NA values for a column with the word "missing" so that they won't be dropped later on"""
        for col in cols:
            self.df[col].fillna('missing', inplace=True)

    def drop_all_non_numeric(self):
        #self.df = self.df.head(1000)
        self.df = self.df[['fraud', 'listed', 'name_length', 'has_header', 'total_cost', 'body_length', 'num_payouts', 'org_twitter', 'has_org_name', 'has_analytics', 'has_logo', 'org_facebook', 'has_payee_name']]
        #self.df.drop(['approx_payout_date', 'country',  ])

    def get_text(self, raw_html):
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text()

    def add_plaintext(self):
        self.df['text_description'] = self.df['description'].apply(self.get_text)

    def assign_text_cluster(self):
        self.add_plaintext()

    def div_count_pos_neg(self, X, y):
        """Helper function to divide X & y into positive and negative classes
        and counts up the number in each.

        Parameters
        ----------
        X : ndarray - 2D
        y : ndarray - 1D

        Returns
        -------
        negative_count : Int
        positive_count : Int
        X_positives    : ndarray - 2D
        X_negatives    : ndarray - 2D
        y_positives    : ndarray - 1D
        y_negatives    : ndarray - 1D
        """
        negatives, positives = y == 0, y == 1
        negative_count, positive_count = np.sum(negatives), np.sum(positives)
        X_positives, y_positives = X[positives], y[positives]
        X_negatives, y_negatives = X[negatives], y[negatives]
        return negative_count, positive_count, X_positives, \
               X_negatives, y_positives, y_negatives

    def oversample(self, X, y, tp):
       """Randomly choose positive observations from X & y, with replacement
       to achieve the target proportion of positive to negative observations.

       Parameters
       ----------
       X  : ndarray - 2D
       y  : ndarray - 1D
       tp : float - range [0, 1], target proportion of positive class observations

       Returns
       -------
       X_undersampled : ndarray - 2D
       y_undersampled : ndarray - 1D
       """
       if tp < np.mean(y):
           return X, y
       neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = self.div_count_pos_neg(X, y)
       positive_range = np.arange(pos_count)
       positive_size = (tp * neg_count) / (1 - tp)
       positive_idxs = np.random.choice(a=positive_range,
                                        size=int(positive_size),
                                        replace=True)
       X_positive_oversampled = X_pos[positive_idxs]
       y_positive_oversampled = y_pos[positive_idxs]
       X_oversampled = np.vstack((X_positive_oversampled, X_neg))
       y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

       return X_oversampled, y_oversampled

    def drop_some_cols(self, columns):
        for col in columns:
            self.df = self.df.drop(col,axis=1)

    def fix_listed(self):
        self.df['listed'] = self.df['listed'].astype(str)
        d = {'y':1,'n':0}
        self.df['listed'] = self.df['listed'].map(d)

    def make_previous_payouts_total(self):
        self.df['num_previous_payouts'] = self.df['previous_payouts'].apply(len)

    def make_total_ticket_cost(self):
        total_cost = []
        for row in self.df['ticket_types']:
            cost = 0
            for i in range(len(row)):
                cost += row[i]['cost']
            total_cost.append(cost)
        self.df['total_cost'] = total_cost

    def make_num_ticket_types(self):
        self.df['num_ticket_types'] = self.df['ticket_types'].apply(len)

    def have_or_not(self,columns):
        """Fill in missing columns / whitespace with 'nan', then create new column to indicate if event has column value or not"""
        #org name, payee name
        for column in columns:
            self.df[column] = self.df[column].replace('',np.nan)
            self.df['has_'+str(column)] = self.df[column].notnull().astype(int)

    def fix_have_header(self):
        self.mark_missing(['has_header'])
        self.dummify(['has_header'])

    def zero_versus_rest(self, columns):
        for col in columns:
            self.df[col+"_is_0"] = self.df[col]==0.0
            self.df[col+"_not_0"] = self.df[col]!=0.0
            self.df = self.df.drop(col, axis=1)




    def clean(self, regression=False, predict=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.fix_listed()
        self.make_previous_payouts_total()
        self.make_total_ticket_cost()
        self.make_num_ticket_types()
        self.have_or_not(['org_name','payee_name', 'payout_type'])
        #self.drop_all_non_numeric()
        self.fix_have_header()
        self.zero_versus_rest(['org_facebook', 'org_twitter', 'channels', 'delivery_method', 'num_order', 'num_payouts', 'user_age'])
        todrop = ['acct_type', 'currency', 'description', 'email_domain', 'event_created','event_end', 'event_published', 'event_start', 'fb_published','gts', 'has_analytics', 'has_logo', 'name','object_id','payee_name',  'previous_payouts','sale_duration', 'sale_duration2', 'show_map', 'ticket_types',\
            'user_created', 'user_type', 'venue_address', 'venue_country','venue_latitude', 'venue_longitude', 'venue_name','venue_state', 'org_desc', 'org_name', 'country', 'payout_type']
        self.drop_some_cols(todrop)




        #import ipdb; ipdb.set_trace()

        #self.drop_na()

        #self.assign_text_cluster()

        y = self.df.pop('fraud').values

        if regression:
            self.drop_columns_for_regression()
            for col in ['avg_dist', 'avg_rating_by_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct']:
                self.df[col] = scale(self.df[col])

        X = self.df.values
        if not predict:
            X_oversampled, y_oversampled = self.oversample(X, y, tp=0.3)
            return X_oversampled, y_oversampled

        return X, y
