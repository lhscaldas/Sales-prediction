import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from datetime import datetime, timedelta


from sklearn.preprocessing import LabelEncoder

class Preprocessor():
    def __init__(self):

        self.train = None
        self.holiday_train = None
        self.test = None
        self.holiday_test = None
        self.dataset = None
        

    def import_train(self):
        self.train = pd.read_csv(
            'train.csv',
            usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
            dtype={
                'store_nbr': 'category',
                'family': 'category',
                'sales': 'float32',
                'onpromotion': 'uint32',
            },
            parse_dates=['date'],
        )
        self.train['date'] = self.train.date.dt.to_period('D')
        self.train['item'] = "S" + self.train['store_nbr'].astype(str) + "_" + self.train['family'].astype(str)
        self.dataset = self.train.pivot(index='date', columns='item', values='sales')
        self.train=self.train.set_index(['date']).sort_index()

    def import_test(self):
        self.test = pd.read_csv(
            'test.csv',
            usecols=['store_nbr', 'family', 'date', 'onpromotion'],
            dtype={
                'store_nbr': 'category',
                'family': 'category',
                'onpromotion': 'uint32',
            },
            parse_dates=['date'],
        )
        self.test['date'] = self.test.date.dt.to_period('D')
        self.test['item'] = "S" + self.test['store_nbr'].astype(str) + "_" + self.test['family'].astype(str)
        self.test=self.test.set_index(['date']).sort_index()
    
    def import_holidays(self):
        holidays_events = pd.read_csv(
            "holidays_events.csv",
            dtype={
                'type': 'category',
                'locale': 'category',
                'locale_name': 'category',
                'description': 'category',
                'transferred': 'bool',
            },
            parse_dates=['date'],
        )       
        holidays_events = holidays_events.set_index('date').to_period('D')
        holidays = (
            holidays_events
            .query("locale in ['National', 'Regional']")
            .loc['2016':'2017-08-16', ['description']]
            .assign(description=lambda x: x.description.cat.remove_unused_categories())
        )
        holidays = pd.get_dummies(holidays, dtype=float)
        self.holidays_train, self.holidays_test = holidays[:"2017-08-01"], holidays["2017-08-01":"2017-08-15"]

    def fill_christmas(self):
        missing_dates = ['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25']
        date_objects = pd.to_datetime(missing_dates, format='%Y-%m-%d')
        missing_df = pd.DataFrame({'date': date_objects})
        missing_df['date'] = missing_df.date.dt.to_period('D')
        missing_df = missing_df.set_index(['date']).sort_index()
        self.dataset = pd.concat([self.dataset, missing_df])
        self.dataset.fillna(0, inplace=True)
        self.dataset.sort_index(inplace=True)
        self.dataset = self.dataset.rename_axis("item", axis="columns")

    def find_zeros(self):
        zero_sales = list(self.dataset.loc[:, (self.dataset == 0).all()].columns)
        self.train['available'] = np.where(self.train['item'].isin(zero_sales), 0, 1)
        self.test['available'] = np.where(self.test['item'].isin(zero_sales), 0, 1)


    def generate_dataset(self):
        self.import_train()
        self.import_test()
        self.import_holidays()
        self.fill_christmas()
        self.find_zeros()

    def fourier(self, y):
        fourier = CalendarFourier(freq='M', order=4)
        dp = DeterministicProcess(
            index=y.index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        return dp

    def generate_training_data(self, START, END):
        y = self.dataset.loc[START:END]
        # X1: Features for Linear Regression
        dp = self.fourier(y)
        X1 = dp.in_sample()
        X1['NewYear'] = (X1.index.dayofyear == 1)
        # X2: Features for XGBoost
        X2 = self.train.drop(['sales', 'store_nbr', 'item'], axis=1).loc[START:END]  # sobra apenas onpromotion
        le = LabelEncoder()
        le.fit(X2.family)
        X2['family'] = le.transform(X2['family'])
        X2["day"] = X2.index.day 
        X2 = X2.join(self.holidays_train, on='date').fillna(0.0)
        return  y, X1, X2
    
    def train_valid_split(self, y, X1, X2, split_date):
        split_datetime = datetime.strptime(split_date, "%Y-%m-%d")
        split_datetime2 = split_datetime + timedelta(days=1)
        split_date2 = split_datetime2.strftime("%Y-%m-%d")

        y_train, y_valid = y[:split_date], y[split_date2:]
        X1_train, X1_valid = X1[: split_date], X1[split_date2 :]
        X2_train, X2_valid = X2.loc[:split_date], X2.loc[split_date2:]
        return y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid

    
    def generate_test_data(self, START, END):
        y = self.dataset.loc[START:END]
        # X1: Features for Linear Regression
        dp = self.fourier(y)
        X1 = dp.out_of_sample(steps=16)
        X1['NewYear'] = (X1.index.dayofyear == 1)
        # X2: Features for XGBoost
        X2 = self.test.drop(['store_nbr', 'item'], axis=1).loc[START:END]  # sobra apenas onpromotion
        le = LabelEncoder()
        le.fit(X2.family)
        X2['family'] = le.transform(X2['family'])
        X2["day"] = X2.index.day 
        X2 = X2.join(self.holidays_test, on='date').fillna(0.0)
        return  X1, X2
    
if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y, X1, X2 = preprocessor.generate_training_data('2017','2017')
    y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid = preprocessor.train_valid_split(y, X1, X2, "2017-07-01")
    print(preprocessor.dataset.head())
    print(preprocessor.dataset.tail())

    

