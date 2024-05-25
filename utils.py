import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
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
    ax.set_title("Periodogram")
    return ax

# Classe para Análise Exploratória dos Dados
class EDA:
    def __init__(self, name):
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

    def family_pivot(self):
        df = self.dataset.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot_table(index='date', columns='family', values='sales', aggfunc='sum', fill_value=0)
        df.reset_index(inplace=True)
        return df

    def initial_exploitation(self):
        fig, ax = plt.subplots(3,1)
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
        plt.show()

    def data_description(self):
        df = self.family_pivot()
        print(df.select_dtypes(include=['number']).describe())
        colunas = df.select_dtypes(include=['number']).columns.tolist()
        # Número de colunas por figura
        familias_por_fig = 3  # Reduzido para melhorar a visualização
        # Número de figuras necessárias
        num_figs = (len(colunas) + familias_por_fig - 1) // familias_por_fig
        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(nrows=familias_por_fig, ncols=2, figsize=(15, familias_por_fig * 5))
            for i in range(familias_por_fig):
                col_idx = fig_idx * familias_por_fig + i
                if col_idx >= len(colunas):
                    break
                coluna = colunas[col_idx]
                # Boxplot
                df[coluna].plot(kind='box', ax=axs[i, 0])
                axs[i, 0].set_title(f'Boxplot de {coluna}')
                axs[i, 0].set_xlabel('')
                axs[i, 0].set_ylabel('')
                # Histograma
                axs[i, 1].hist(df[coluna], bins=10, density=True, alpha=0.7)
                axs[i, 1].set_title(f'Histograma de {coluna}')
                axs[i, 1].set_xlabel('')
                axs[i, 1].set_ylabel('')
            fig.tight_layout(pad=3.0)
        # Mostrar todas as figuras
        plt.show()
    
    def plot_sales_by_family(self):
        df = self.family_pivot()
        familias_por_fig=3
        colunas = df.select_dtypes(include=['number']).columns.tolist()
        # Número de figuras necessárias
        num_figs = (len(colunas) + familias_por_fig - 1) // familias_por_fig
        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(nrows=familias_por_fig, ncols=1, figsize=(15, familias_por_fig * 5))
            for i in range(familias_por_fig):
                col_idx = fig_idx * familias_por_fig + i
                if col_idx >= len(colunas):
                    break
                coluna = colunas[col_idx]
                trend365 =  df[coluna].rolling(
                        window=365,
                        center=True,
                        min_periods=183,
                    ).mean()
                trend30 =  df[coluna].rolling(
                        window=30,
                        center=True,
                        min_periods=15,
                    ).mean()
                # Gráfico de linha para vendas
                axs[i].plot(df['date'], df[coluna], marker='.', markerfacecolor='white', markeredgewidth=1, 
                            linestyle='-', color='0.25', label='Vendas', linewidth=1, markersize=4)
                # Gráfico de linha para tendência
                axs[i].plot(df['date'], trend365, color='blue', label='Média móvel (365 dias)')
                axs[i].plot(df['date'], trend30, color='red', label='Média móvel (30 dias)')
                axs[i].set_title(f'{coluna}')
                axs[i].set_xlabel('date')
                axs[i].set_ylabel(coluna)
                axs[i].legend()
            fig.tight_layout(pad=3.0)
        # Mostrar todas as figuras
        plt.show()

    def plot_periodogram_by_family(self):
        df = self.family_pivot()
        familias_por_fig=6
        colunas = df.select_dtypes(include=['number']).columns.tolist()
        # Número de figuras necessárias
        num_figs = (len(colunas) + familias_por_fig - 1) // familias_por_fig
        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
            for i in range(3):
                for j in range(2):
                    fam_idx = fig_idx * 6 + i * 2 + j
                    if fam_idx >= len(colunas):
                        break
                    coluna = colunas[fam_idx]
                    # Calcular o periodograma
                    f, Pxx = periodogram(
                        df[coluna],
                        fs=pd.Timedelta("365D") / pd.Timedelta("1D"),
                        detrend='linear',
                        window="boxcar",
                        scaling='spectrum',
                    )
                    # Plotar o periodograma
                    axs[i, j].step(f, Pxx, color="purple")
                    axs[i, j].set_xscale("log")
                    axs[i, j].set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
                    axs[i, j].set_xticklabels(
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
                    axs[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    axs[i, j].set_ylabel("Variance")
                    axs[i, j].set_title(f'Periodograma de {coluna}')
            fig.tight_layout(pad=3.0)
        # Mostrar a figura atual
        plt.show()


    



        