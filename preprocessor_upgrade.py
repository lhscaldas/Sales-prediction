import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from datetime import date, datetime, timedelta
import time
import calendar

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor



class Preprocessor():
    def __init__(self):
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.holiday = pd.read_csv('holidays_events.csv')
        self.sample = pd.read_csv('sample_submission.csv')
        self.oil = pd.read_csv('oil.csv')
        self.stores = pd.read_csv('stores.csv')
        self.transactions = pd.read_csv('transactions.csv')

    def gen_series(self):
        df = self.train.copy()
        df['item'] = "S" + df['store_nbr'].astype(str) + "_" + df['family'].astype(str)
        self.series = df.pivot(index='date', columns='item', values='sales')

    def preprocess_traintest(self, df):
        df['date'] = df['date'].map(lambda x: date.fromisoformat(x))
        df['weekday'] = df['date'].map(lambda x: x.weekday())
        df['year'] = df['date'].map(lambda x: x.year)
        df['month'] = df['date'].map(lambda x: x.month)
        df['day'] = df['date'].map(lambda x: x.day)
        df['eomd'] = df['date'].map(lambda x: calendar.monthrange(x.year, x.month)[1])
        df['payday'] = ((df['day'] == df['eomd'])|(df['day'] == 15)).astype(int)
        df.drop(['id', 'eomd'], axis=1, inplace=True)
        return df
    
    def preprocess_oil(self):
        oil = self.oil.copy()
        oil['month'] = oil['date'].map(lambda x: int(x.replace('-', '')[:6]))
        oil['month_avg'] = oil.groupby('month')['dcoilwtico'].transform('mean')
        oil['tmp'] = oil['dcoilwtico'].map(np.isnan)
        oil['month_avg'] = oil['tmp'] * oil['month_avg']
        oil['dcoilwtico'].fillna(0, inplace=True)
        oil['dcoilwtico'] = oil['dcoilwtico'] + oil['month_avg']
        oil['dcoilwtico'] = oil['dcoilwtico'].astype(float)
        oil = oil.drop(['month', 'month_avg', 'tmp'], axis=1)
        oil['date'] = oil['date'].map(lambda x: date.fromisoformat(x))
        self.oil = oil

    def preprocess_holiday(self):
        # separar holiday de event e earthquake
        holiday = self.holiday.copy()
        holiday['date'] = holiday['date'].map(lambda x: date.fromisoformat(x))
        holiday = holiday[(holiday['transferred']==False)&(holiday['type']!='Work Day')]
        event = holiday[holiday['type']=='Event']
        earthquake = event[event['description'].str.startswith('Terremoto Manabi')]
        event = event[event['description'].str.startswith('Terremoto Manabi')==False]
        # arrumar os dados de event e earthquake
        event = event[['date', 'description']]
        event.rename({'description': 'event_name'}, axis=1, inplace=True)
        earthquake = earthquake[['date', 'description']]
        earthquake.rename({'description': 'earthquake'}, axis=1, inplace=True)
        # separar os tipos de feriado
        h_local = holiday[holiday['locale']=='Local']
        h_local = h_local[['date', 'locale_name', 'description']]
        h_local = h_local.rename({'locale_name': 'city', 'description': 'local_holiday_name'}, axis=1)
        h_regional = holiday[holiday['locale']=='Regional']
        h_regional = h_regional[['date', 'locale_name', 'description']]
        h_regional = h_regional.rename({'locale_name': 'state', 'description': 'regional_holiday_name'}, axis=1)
        h_national = holiday[holiday['locale']=='National']
        h_national = h_national[['date', 'description']]
        h_national = h_national.rename({'description': 'national_holiday_name'}, axis=1)
        return event, earthquake, h_local, h_regional, h_national
    
    def merge_tables(self, df):
        event, earthquake, h_local, h_regional, h_national = self.preprocess_holiday()
        df = df.merge(self.oil, on='date', how='left')
        df = df.merge(self.stores, on='store_nbr', how='left')
        df = df.merge(event, on='date', how='left').fillna('0')
        df = df.merge(earthquake, on='date', how='left').fillna('0')
        df = df.merge(h_local, on=['date', 'city'], how='left').fillna('0')
        df = df.merge(h_regional, on=['date', 'state'], how='left').fillna('0')
        df = df.merge(h_national, on='date', how='left').fillna('0')
        df = df.merge(self.transactions, on=['date', 'store_nbr'], how='left').fillna(0)
        return df
    
    def label_encoding(self):
        cat_features = ['family', 'store_nbr', 'city', 'state', 'type', 'cluster',
                'event_name', 'earthquake', 'local_holiday_name', 'regional_holiday_name', 'national_holiday_name']
        for col in cat_features:
            le = LabelEncoder()
            self.train[col] = le.fit_transform(self.train[col])
            self.test[col] = le.transform(self.test[col])

    def generate_dataset(self):
        self.gen_series()
        self.train = self.preprocess_traintest(self.train)
        self.test = self.preprocess_traintest(self.test)
        self.preprocess_oil()
        self.train = self.merge_tables(self.train)
        self.test = self.merge_tables(self.test)     

    def fourier(self):
        y = self.series.copy()
        y.index = pd.to_datetime(y.index)
        y.index = y.index.to_period('D')
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
    
    def date_split(self, df, split_date):
        split_datetime = datetime.strptime(split_date, "%Y-%m-%d")
        split_datetime2 = split_datetime + timedelta(days=1)
        split_date2 = split_datetime2.strftime("%Y-%m-%d")
        if 'date' in df.columns.tolist():       
            df_train = df[df['date']<=split_datetime.date()]
            df_valid = df[df['date']>=split_datetime2.date()]
        else:
            df_train, df_valid = df.loc[:split_date], df.loc[split_date2:]
        return df_train, df_valid

    def generate_training_data(self):
        # y1 e X1: Features for Linear Regression
        y1 = self.series.copy()
        dp = self.fourier()
        X1 = dp.in_sample()
        # y2 e X2: Features for XGBoost
        X2 = self.train.drop(['sales', 'store_nbr','family'], axis=1)
        y2 = self.train.copy()
        y2 = y2[['date','sales']]
        return  y1, y2, X1, X2
    
    def generate_test_data(self):
        # X1: Features for Linear Regression
        dp = self.fourier()
        X1 = dp.out_of_sample(steps=16)
        # X2: Features for XGBoost
        X2 = self.test.drop(['store_nbr','family'], axis=1)
        return  X1, X2


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y1, y2, X1, X2 = preprocessor.generate_training_data()
    # print(y1.shape)
    # print(X1.shape)
    # print(y2.shape)
    # print(X2.shape)
    # y_train, y_valid = preprocessor.date_split(y, "2017-07-01")
    # X1_train, X1_valid = preprocessor.date_split(X1, "2017-07-01")
    # X2_train, X2_valid = preprocessor.date_split(X2, "2017-07-01")
    # print(y_valid.head())
    # print(X1_valid.head())
    # print(X2_valid.head())

    # model = BoostedHybrid(
    #                     model_1=LinearRegression(),
    #                     model_2=XGBRegressor(
    #                         n_estimators=100,
    #                         learning_rate=0.01,
    #                         max_depth=3,
    #                     )
    #                 )
    # model.fit(X1_train, X2_train, y_train)
    # y_fit = model.predict(X1_train, X2_train)
    # y_pred = model.predict(X1_valid, X2_valid)
    # model.validation_results(y_train, y_valid, y_fit, y_pred)