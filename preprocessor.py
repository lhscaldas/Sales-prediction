import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier


from sklearn.preprocessing import LabelEncoder

class Preprocessor():
    def __init__(self):
        # self.oil = pd.read_csv('oil.csv')
        # self.stores = pd.read_csv('stores.csv')
        # self.transactions = pd.read_csv('transactions.csv')

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
        fourier = CalendarFourier(freq='ME', order=4)
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
        # X_1: Features for Linear Regression
        dp = self.fourier(y)
        X_1 = dp.in_sample()
        X_1['NewYear'] = (X_1.index.dayofyear == 1)
        # X_2: Features for XGBoost
        X_2 = self.train.drop(['sales', 'store_nbr', 'item'], axis=1).loc[START:END]  # sobra apenas onpromotion
        le = LabelEncoder()
        le.fit(X_2.family)
        X_2['family'] = le.transform(X_2['family'])
        X_2["day"] = X_2.index.day 
        X_2 = X_2.join(self.holidays_train, on='date').fillna(0.0)
        y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
        X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
        X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]
        return  y, y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid
    
    def generate_test_data(self, START, END):
        y = self.dataset.loc[START:END]
        # X_1: Features for Linear Regression
        dp = self.fourier(y)
        X_1 = dp.out_of_sample(steps=16)
        X_1['NewYear'] = (X_1.index.dayofyear == 1)
        # X_2: Features for XGBoost
        X_2 = self.test.drop(['store_nbr', 'item'], axis=1).loc[START:END]  # sobra apenas onpromotion
        le = LabelEncoder()
        le.fit(X_2.family)
        X_2['family'] = le.transform(X_2['family'])
        X_2["day"] = X_2.index.day 
        X_2 = X_2.join(self.holidays_test, on='date').fillna(0.0)
        return  X_1, X_2
    

