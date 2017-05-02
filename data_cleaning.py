import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

class DataCleaning(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, filepath):
        """Reads in data

        Args:
            fileapth (str): location of file with csv data
        """
        self.df = pd.read_csv(filepath)


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
        self.df = self.df.head(1000)
        self.df = self.df[['fraud', 'body_length', 'channels', 'num_payouts', 'org_twitter']]
        #self.df.drop(['approx_payout_date', 'country',  ])

    def clean(self, regression=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.drop_all_non_numeric()
        self.drop_na()

        y = self.df.pop('fraud').values

        if regression:
            self.drop_columns_for_regression()
            for col in ['avg_dist', 'avg_rating_by_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct']:
                self.df[col] = scale(self.df[col])

        X = self.df.values

        return X, y
