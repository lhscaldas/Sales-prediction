from utils import EDA

eda = EDA('train.csv')
eda.family_pivot()
eda.initial_exploitation()
eda.data_description()
eda.plot_sales_by_family()
eda.plot_periodogram_by_family()