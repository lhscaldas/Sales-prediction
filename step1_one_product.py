from utils import EDA

eda = EDA('train.csv')
familia = 'AUTOMOTIVE'
# eda.family_analysis(familia)
eda.family_deseason(familia, order = 4)