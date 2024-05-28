import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

from sklearn.linear_model import LinearRegression

# Classe para Análise Exploratória dos Dados
class EDA:
    def __init__(self, name, initial=None):
        self.dataset = pd.read_csv(
            name,
            usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
            dtype={
                'store_nbr': 'category',
                'family': 'category',
                'sales': 'float32',
                'onpromotion': 'uint32',
            },
            parse_dates=['date'],
        )
        self.dataset['date'] = self.dataset.date.dt.to_period('D')

        if initial:
            self.dataset=self.dataset[self.dataset['date'] >= initial]

        self.family_sales = (
             self.dataset
             .drop('onpromotion', axis=1)
             .set_index(['store_nbr', 'family', 'date'])
             .sort_index()
             .groupby(['family', 'date'], observed=False)
             .sum()
             .unstack('family')
             .fillna(0)
             )

    def initial_exploitation(self):
        fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(16, 8))
        df = [self.dataset.head(),self.dataset.tail()]
        for i in range(2):
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
            ax[i].set_frame_on(False)
            table = ax[i].table(cellText=df[i].values, colLabels=df[i].columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)
        ax[0].set_title("head")
        ax[1].set_title("tail")
        texto = f"{self.dataset.isnull().any().sum()} dados faltantes no dataset de treino \n"
        texto += f"{self.dataset.duplicated().sum()} linhas duplicadas no dataset de treino \n"
        texto += f"shape {self.dataset.shape} \n"
        cols_categoricas = self.dataset.select_dtypes(include=['category']).nunique()
        texto += f'{cols_categoricas.shape[0]} colunas de dados categóricos \n'
        num_family= cols_categoricas['family']
        texto += f'{num_family} familias diferentes \n'
        ax[2].text(0.5, 0.5, texto, fontsize=12, ha='center', va='center', wrap=True)
        ax[2].xaxis.set_visible(False)
        ax[2].yaxis.set_visible(False)
        ax[2].set_frame_on(False)
        plt.tight_layout()
        plt.show()

    def plot_periodogram(self, y, familia, ax):
        # Periodograma
        f, Pxx = periodogram(
            y.values.reshape(-1),
            fs=pd.Timedelta("365D") / pd.Timedelta("1D"),
            detrend='linear',
            window="boxcar",
            scaling='spectrum',
        )
        # return print(y, f, Pxx), print(type(y), type(f), type(Pxx))
        ax.step(f, Pxx, color="purple")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
            ],
            rotation=30,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Variance")
        ax.set_title(f'Periodograma de {familia}')
        return ax

    def family_analysis(self, familia):
        df = self.family_sales.loc(axis=1)[:, familia]
        df.index = df.index.to_timestamp()
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        # Plot sales e trend
        ax0 = fig.add_subplot(gs[0, :])
        trend365 =  df['sales'].rolling(
                        window=365,
                        center=True,
                        min_periods=183,
                    ).mean()
        trend30 =  df['sales'].rolling(
                        window=30,
                        center=True,
                        min_periods=15,
                    ).mean()
        ax0.plot(df.index, df['sales'], marker='.', linestyle='-', color='0.25', label='vendas', linewidth=1, markersize=4)
        ax0.plot(df.index, trend365, color='blue', label='média móvel (365 dias)')
        ax0.plot(df.index, trend30, color='red', label='média móvel (30 dias)')
        ax0.set_title(f'Vendas e tendências para {familia}')
        ax0.set_xlabel('Date')
        ax0.set_ylabel(familia)
        ax0.legend()
        # Boxplot e Histograma
        ax1 = fig.add_subplot(gs[1, 0])
        df['sales'].plot(kind='box', ax=ax1)
        ax1.set_title(f'Boxplot de {familia}')
        ax1.set_xlabel('')
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.hist( df['sales'], bins=10, density=True, alpha=0.7)
        ax2.set_title(f'Histograma de {familia}')
        ax2.set_xlabel(familia)
        ax2.set_ylabel('Density')
        # Periodograma
        ax3 = fig.add_subplot(gs[2, :])
        ax3 = self.plot_periodogram(df['sales'],familia,ax3)
        plt.tight_layout()
        plt.show()

    def all_families_analysis(self):
        df = self.family_sales
        familias = df.select_dtypes(include=['number']).columns.tolist()
        for familia in familias:
            self.family_analysis(familia[1])
       
    def calc_deseason(self, familia, freq, order):
        sales = self.family_sales.loc(axis=1)[:, familia]
        y = sales.loc[:, 'sales'].squeeze()
        # vendas e sua sazonalidade
        fourier = CalendarFourier(freq=freq, order=order)
        dp = DeterministicProcess(
            index=y.index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        X = dp.in_sample()
        model = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        y_dessaz = pd.Series(y.values - y_pred.values, index=X.index)
        return y, y_pred, y_dessaz

    def family_deseason(self, familia, freq, order):
        y, y_pred, y_dessaz = self.calc_deseason(familia, freq, order)
        _, ax = plt.subplots(nrows=3, figsize=(16, 8))
        ax[0].plot(y.index.to_timestamp(), y.values, marker='.', linestyle='-', color='0.25', label='vendas', linewidth=1, markersize=4)
        ax[0].plot(y.index.to_timestamp(), y_pred, label='sazonalidade')
        ax[0].plot(y.index.to_timestamp(), y_dessaz, label='dessazonalizado')
        ax[0].set_title(f'Vendas e sazonalidade para {familia}')
        ax[0].set_ylabel('vendas')
        ax[0].legend()
        # periodorama sazonal
        ax[1] = self.plot_periodogram(y, familia, ax[1])
        ax[1].set_title(f'Periodograma de {familia} sazonal')
        # periodorama dessazonalizado
        ax[2] = self.plot_periodogram(y_dessaz, familia, ax[2])
        ax[2].set_title(f'Periodograma de {familia} dessazonalizado')
        y_lim = ax[1].get_ylim()
        ax[2].set_ylim(top=y_lim[1])
        plt.tight_layout()
        plt.show()

    def all_families_deseason(self, freq, order):
        df = self.family_sales
        familias = df.select_dtypes(include=['number']).columns.tolist()
        for familia in familias:
            self.family_deseason(familia[1], freq, order)

    def family_lag(self, familia, freq, order, max_lag=8):
        _, _, y_dessaz = self.calc_deseason(familia, freq, order)
        df = y_dessaz.to_frame(name='sales')
        # Gerando as colunas de lag
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df['sales'].shift(lag)

        # Definindo o número de colunas e linhas para os subplots
        num_cols = 4
        num_rows = (max_lag + 1) // num_cols + ((max_lag + 1) % num_cols > 0) 
        print(num_rows)
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(num_rows, num_cols, wspace=0.2, hspace=0.6)
        # Plotando os gráficos de dispersão
        for lag in range(1, max_lag + 1):
            correlation = df['sales'].corr(df[f'lag_{lag}'])
            row_idx = (lag-1) // num_cols  # Índice da linha para o subplot
            col_idx = (lag-1) % num_cols  # Índice da coluna para o subplot
            ax = fig.add_subplot(gs[row_idx, col_idx]) if num_rows > 1 else fig.add_subplot(gs[col_idx])
            sns.scatterplot(x=f'lag_{lag}', y='sales', data=df, ax=ax)
            sns.regplot(x=f'lag_{lag}', y='sales', data=df, scatter=False, color='red', ci=None, ax=ax)
            ax.set_title(f'Lag {lag} vs {familia:5}')
            ax.set_xlabel(f'Lag {lag}')
            ax.set_ylabel('Vendas')
            ax.grid(True)
            ax.text(0.05, 0.95, f'r: {correlation:.2f}', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        ax_acf = fig.add_subplot(gs[-1, :])
        plot_pacf(df['sales'], lags=max_lag, ax=ax_acf)
        acf_values = pacf(df['sales'], nlags=max_lag)
        delta = 0.05
        ax_acf.set_ylim(top=max(max(acf_values), 0) + delta, bottom=min(min(acf_values), 0) - delta)
        ax_acf.set_xlabel('Lag')
        ax_acf.set_ylabel('Autocorrelation')
        ax_acf.grid(True)
        plt.show()

    def all_families_lag(self, freq, order, max_lag=8):
        df = self.family_sales
        familias = df.select_dtypes(include=['number']).columns.tolist()
        for familia in familias:
            self.family_lag(familia[1], freq, order, max_lag)



# if __name__ == '__main__':
    # eda = EDA('train.csv')
    # eda.initial_exploitation()
    # eda.all_families_analysis()

    # eda = EDA('train.csv', initial = '2016-08-15')
    # eda.all_families_deseason('A', 26)
    # eda.all_families_lag(freq='M', order=4, max_lag=8)

    # familia = 'SCHOOL AND OFFICE SUPPLIES'
    # eda = EDA('train.csv', initial = '2017-01-01')
    # eda.family_analysis(familia)
    # eda.family_deseason(familia, freq='M', order = 4)
    # eda.family_lag(familia, freq='M', order=4, max_lag=8)



