from preprocessor_upgrade import Preprocessor
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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
        y_resid = y.subtract(y_fit, axis='columns', fill_value=0)
        y_resid = y_resid.stack().squeeze()
        X_2 = X_2.drop(['date'], axis=1)
        self.model_2.fit(X_2, y_resid.values)
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()
        X_2 = X_2.drop(['date'], axis=1)
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


if __name__ == '__main__':
    # Pré-processamento
    preprocessor = Preprocessor()
    preprocessor.generate_dataset()
    y, X1, X2 = preprocessor.generate_training_data()
    y_train, y_valid = preprocessor.date_split(y, "2017-07-01")
    X1_train, X1_valid = preprocessor.date_split(X1, "2017-07-01")
    X2_train, X2_valid = preprocessor.date_split(X2, "2017-07-01")
    # Modelo
    xgb_params = {
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse', 
        'learning_rate': 0.01,
        'subsample': 0.99,
        'colsample_bytree': 0.80,
        'reg_alpha': 10.0,
        'reg_lambda': 0.18,
        'min_child_weight': 47,
    }
    model = BoostedHybrid(
                        model_1=LinearRegression(),
                        model_2=XGBRegressor(**xgb_params)
                    )
    model.fit(X1_train, X2_train, y_train)
    y_fit = model.predict(X1_train, X2_train)
    # Validação
    y_pred = model.predict(X1_valid, X2_valid)
    model.validation_results(y_train, y_valid, y_fit, y_pred)
    # Teste
    model.fit(X1, X2, y)
    X1_test, X2_test = preprocessor.generate_test_data()
    y_test = model.predict(X1_test, X2_test)
    model.generate_csv(y_test)