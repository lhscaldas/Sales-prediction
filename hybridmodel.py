import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessor import Preprocessor

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

class BoostedHybrid:
    def __init__(self, model_1=Ridge(), model_2=XGBRegressor()):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None
        self.y_fit = None
        self.y_resid = None

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1), 
            index=X_1.index, columns=y.columns,
        )
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze()
        self.model_2.fit(X_2, y_resid)
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()
        y_pred += self.model_2.predict(X_2)
        return y_pred.unstack()
    
    def validation_results(self, y_train, y_valid, y_fit, y_pred, verbose=True, plot=True):
            y_fit = y_fit.clip(0.0)
            y_pred = y_pred.clip(0.0)
            error_train = np.sqrt(mean_squared_log_error(y_train, y_fit))
            error_valid = np.sqrt(mean_squared_log_error(y_valid, y_pred))
            if verbose:
                print(f"erro de treinamento: {error_train}")
                print(f"erro de validação: {error_valid}")
            if plot:
                y = pd.concat([y_train, y_valid], axis=0)
                families = y.columns[0:6]
                axs = y.loc(axis=1)[families].plot(
                    subplots=True, figsize=(11, 9), marker='.', linestyle='-',
                    color='0.25', label='vendas', linewidth=1, markersize=4, alpha=0.5)
                _ = y_fit.loc(axis=1)[families].plot(subplots=True, color='C0', ax=axs)
                _ = y_pred.loc(axis=1)[families].plot(subplots=True, color='C3', ax=axs)
                for ax, family in zip(axs, families):
                    ax.legend([])
                    ax.set_ylabel(family)
                plt.show()
            return error_train, error_valid
    
    def generate_csv(self, y_test):
        y_test = y_test.reset_index(drop=False)
        y_test = y_test.rename(columns={'index': 'date'})
        print(y_test.columns)
        y_test['date'] = y_test['date'].astype(str)
        unpivoted = (y_test
                        .melt(id_vars='date', var_name='item', value_name='sales')
                        )
        unpivoted[['store_nbr', 'family']] = unpivoted['item'].str.extract(r'S(\d+)_(.+)')
        unpivoted = (unpivoted
            .dropna()
            .drop(['item'], axis=1)
        )
        output = pd.read_csv(
                    'test.csv',
                    usecols=['id','store_nbr', 'family', 'date', 'onpromotion'],
                    dtype={
                        'store_nbr': 'category',
                        'family': 'category',
                        'onpromotion': 'uint32',
                    },
                    parse_dates=['date'],
                )
        output['date'] = output['date'].astype(str)
        output = output.merge(unpivoted, on=['date', 'store_nbr', 'family'], how='left', suffixes=('', '_unpivoted'))
        output = output.drop(['date', 'store_nbr', 'family','onpromotion'], axis=1)
        output.to_csv('submission.csv',index=False)

def optimize():
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y, X1, X2 = preprocessor.generate_training_data('2017','2017')
    y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid = preprocessor.train_valid_split(y, X1, X2, "2017-07-01")

    model1_list = [LinearRegression(), Lasso(alpha=0.01), Ridge(alpha=0.01), Lasso(alpha=0.1), Ridge(alpha=0.1)]
    n_estimators_list = [100, 500, 1000]
    learning_rate_list = [0.01, 0.05, 0.1]
    max_depth_list = [3, 6, 9]

    best_model = list()
    best_valid_error = 5
    best_train_error = 5

    for model1 in model1_list:
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    model = BoostedHybrid(
                        model_1=model1,
                        model_2=XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth
                        )
                    )
                    model.fit(X1_train, X2_train, y_train)
                    y_fit = model.predict(X1_train, X2_train)
                    y_pred = model.predict(X1_valid, X2_valid)
                    error_train, error_valid = model.validation_results(y_train, y_valid, y_fit, y_pred, verbose=False)
                    if error_valid < best_valid_error and error_train < error_valid:
                        best_valid_error = error_valid
                        best_train_error = error_train
                        best_model = [model1, n_estimators, learning_rate, max_depth]
                        print(f"melhor modelo: {best_model}")
                        print(f"erro de treinamento: {error_train}")
                        print(f"erro de validação: {error_valid}")

def validate():
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y, X1, X2 = preprocessor.generate_training_data('2017','2017')
    y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid = preprocessor.train_valid_split(y, X1, X2, "2017-07-01")

    model = BoostedHybrid(
                        model_1=LinearRegression(),
                        model_2=XGBRegressor(
                            n_estimators=100,
                            learning_rate=0.01,
                            max_depth=3,
                        )
                    )
    model.fit(X1_train, X2_train, y_train)
    y_fit = model.predict(X1_train, X2_train)
    y_pred = model.predict(X1_valid, X2_valid)
    model.validation_results(y_train, y_valid, y_fit, y_pred)

if __name__ == '__main__':
    validate()
    


    