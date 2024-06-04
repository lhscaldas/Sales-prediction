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
from sklearn.metrics import mean_squared_log_error, mean_squared_error
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

        df['dcoilwtico'] = df['dcoilwtico'].astype(float)
        return df
    
    def label_encoding(self):
        cat_features = ['family', 'store_nbr', 'city', 'state', 'type', 'cluster',
                'event_name', 'earthquake', 'local_holiday_name', 'regional_holiday_name', 'national_holiday_name']
        for col in cat_features:
            try:
                le = LabelEncoder()
                self.train[col] = le.fit_transform(self.train[col])
                self.test[col] = le.transform(self.test[col])
            except Exception as e:
                # Trata qualquer exceção que ocorrer
                print(f"Erro ao processar a coluna '{col}': {e}")

    def date_cut2017(self, df):
        split_date = "2017-01-01"
        split_datetime = datetime.strptime(split_date, "%Y-%m-%d")
        if 'date' in df.columns.tolist():       
            df = df[df['date']>=split_datetime.date()]
        else:
            df = df.loc[split_date:]
        return df

    def generate_dataset(self):
        self.gen_series()
        self.train = self.preprocess_traintest(self.train)
        self.test = self.preprocess_traintest(self.test)
        self.preprocess_oil()
        self.train = self.merge_tables(self.train)
        self.test = self.merge_tables(self.test)
        self.label_encoding()
        self.series = self.date_cut2017(self.series)
        self.train = self.date_cut2017(self.train)

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
        y = self.series.copy()
        # X1: Features for Linear Regression
        dp = self.fourier()
        X1 = dp.in_sample()
        # X2: Features for XGBoost
        X2 = self.train.drop(['sales','store_nbr','family','year'], axis=1)
        return  y, X1, X2
    
    def generate_test_data(self):
        # X1: Features for Linear Regression
        dp = self.fourier()
        X1 = dp.out_of_sample(steps=16)
        # X2: Features for XGBoost
        X2 = self.test.drop(['store_nbr','family','year'], axis=1)
        return  X1, X2


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y, X1, X2 = preprocessor.generate_training_data()
    y_train, y_valid = preprocessor.date_split(y, "2017-07-01")
    X1_train, X1_valid = preprocessor.date_split(X1, "2017-07-01")
    X2_train, X2_valid = preprocessor.date_split(X2, "2017-07-01")

    X = preprocessor.train.drop('sales', axis=1)
    X = X.set_index('date')
    X.index = pd.to_datetime(X.index)
    y = preprocessor.train[['date','sales']]
    y = y.set_index('date')
    y.index = pd.to_datetime(y.index)
    y = np.log(y + 1)
    y_train, y_valid = preprocessor.date_split(y, "2017-07-01")
    X_train, X_valid = preprocessor.date_split(X, "2017-07-01")
    # xgb_params = {
    #     'objective': 'reg:squarederror', 
    #     'eval_metric': 'rmse', 
    #     'learning_rate': 0.01,
    #     'subsample': 0.99,
    #     'colsample_bytree': 0.80,
    #     'reg_alpha': 10.0,
    #     'reg_lambda': 0.18,
    #     'min_child_weight': 47,
    # }
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.01,
    }
    model = XGBRegressor(**xgb_params)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_valid)
    erro_val = np.sqrt(mean_squared_error(y_valid,y_pred))
    print(erro_val)