import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import matplotlib.gridspec as gridspec
from pandas.tseries.offsets import DateOffset
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from sklearn.linear_model import LinearRegression

# Classe para Análise Exploratória dos Dados
class EDA:
    def __init__(self, name, initial=False):
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
        self.dataset['date'] = pd.to_datetime(self.dataset['date'])
        if initial:
            self.dataset = self.dataset[self.dataset['date'] >= initial]
        
        

    def family_pivot(self, familia = False):
        df = self.dataset.copy()
        df = df.pivot_table(index='date', columns='family', values='sales', aggfunc='sum', fill_value=0)
        df.reset_index(inplace=True)
        if familia:
            if familia not in df.columns:
                raise ValueError(f"Family '{familia}' not found in the dataset.")
            df = df[['date', familia]].copy()
        return df

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
            y,
            fs=pd.Timedelta("365D") / pd.Timedelta("1D"),
            detrend='linear',
            window="boxcar",
            scaling='spectrum',
        )
        
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
        df = self.family_pivot(familia = familia)
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        # Plot sales e trend
        ax0 = fig.add_subplot(gs[0, :])
        trend365 =  df[familia].rolling(
                        window=365,
                        center=True,
                        min_periods=183,
                    ).mean()
        trend30 =  df[familia].rolling(
                        window=30,
                        center=True,
                        min_periods=15,
                    ).mean()
        ax0.plot(df['date'], df[familia], marker='.', linestyle='-', color='0.25', label='vendas', linewidth=1, markersize=4)
        ax0.plot(df['date'], trend365, color='blue', label='média móvel (365 dias)')
        ax0.plot(df['date'], trend30, color='red', label='média móvel (30 dias)')
        ax0.set_title(f'Vendas e tendências para {familia}')
        ax0.set_xlabel('Date')
        ax0.set_ylabel(familia)
        ax0.legend()
        # Boxplot e Histograma
        ax1 = fig.add_subplot(gs[1, 0])
        df[familia].plot(kind='box', ax=ax1)
        ax1.set_title(f'Boxplot de {familia}')
        ax1.set_xlabel('')
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.hist(df[familia], bins=10, density=True, alpha=0.7)
        ax2.set_title(f'Histograma de {familia}')
        ax2.set_xlabel(familia)
        ax2.set_ylabel('Density')
        # Periodograma
        ax3 = fig.add_subplot(gs[2, :])
        ax3 = self.plot_periodogram(df[familia],familia,ax3)
        plt.tight_layout()
        plt.show()

    def all_families_analysis(self):
        df = self.family_pivot()
        familias = df.select_dtypes(include=['number']).columns.tolist()
        for familia in familias:
            self.family_analysis(familia)

    def calc_deseason(self, familia, order):
        df = self.family_pivot(familia = familia)
        df.set_index('date', inplace=True)
        df = df.asfreq('D')
        df = df.fillna(df.rolling(3, min_periods=1).mean().shift(-1))
        # vendas e sua sazonalidade
        y = df.copy().squeeze()
        y.dropna(inplace=True)
        fourier = CalendarFourier(freq='A', order=order)
        dp = DeterministicProcess(
            index=y.index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        X = dp.in_sample()
        model = LinearRegression().fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        y_dessaz = pd.Series(y.values - y_pred.values, index=X.index)
        return y, y_pred, y_dessaz

    def family_deseason(self, familia, order):
        y, y_pred, y_dessaz = self.calc_deseason(familia, order)
        fig, ax = plt.subplots(nrows=3, figsize=(16, 8))
        ax[0].plot(y.index, y, marker='.', linestyle='-', color='0.25', label='vendas', linewidth=1, markersize=4)
        ax[0].plot(y.index, y_pred, label='sazonalidade')
        ax[0].plot(y.index, y_dessaz, label='dessazonalizado')
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

    def all_families_deseason(self):
        df = self.family_pivot()
        familias = df.select_dtypes(include=['number']).columns.tolist()
        for familia in familias:
            self.family_deseason(familia, order = 40)

    def family_lag(self, familia, order, max_lag=8):
        _, _, y_dessaz = self.calc_deseason(familia, order)
        df = y_dessaz.to_frame(name='sales')
        # Gerando as colunas de lag
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df['sales'].shift(lag)
        
        # Plotando os gráficos de dispersão
        plt.figure(figsize=(15, 15))
        for lag in range(1, max_lag + 1):
            correlation = df['sales'].corr(df[f'lag_{lag}'])
            plt.subplot(3, 4, lag)
            sns.scatterplot(x=f'lag_{lag}', y='sales', data=df)
            plt.title(f'Lag {lag} vs Vendas de {familia}')
            plt.xlabel(f'Lag {lag}')
            plt.ylabel('Vendas')
            plt.text(0.05, 0.95, f'r: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        ax = plt.subplot(3, 1, 3)
        plot_acf(df['sales'], lags=max_lag, ax=plt.gca())
        acf_values = acf(df['sales'], nlags=max_lag)
        delta = 0.025
        ax.set_ylim(top=max(acf_values+delta), bottom=min(acf_values-delta))
        plt.title('Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        plt.show()


if __name__ == '__main__':
    eda = EDA('train.csv', initial='2016-08-15')
    # eda = EDA('train.csv')
    familia = 'SCHOOL AND OFFICE SUPPLIES'
    # eda.family_analysis(familia)
    order = 4
    eda.family_deseason(familia, order = 4)
    eda.family_lag(familia, order = 8)




