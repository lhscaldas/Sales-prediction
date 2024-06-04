import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Carregar dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
holidays = pd.read_csv('holidays_events.csv')
oil = pd.read_csv('oil.csv')
transactions = pd.read_csv('transactions.csv')
submission = pd.read_csv('sample_submission.csv')

# Mesclar dados para gerar features extras
train = pd.merge(train, holidays, on='date', how='left')
train = pd.merge(train, oil, on='date', how='left')
train = pd.merge(train, transactions, on=['date', 'store_nbr'], how='left')

test = pd.merge(test, holidays, on='date', how='left')
test = pd.merge(test, oil, on='date', how='left')
test = pd.merge(test, transactions, on=['date', 'store_nbr'], how='left')

# Processar dados
le = LabelEncoder()
for col in ['family', 'type', 'locale', 'locale_name', 'description']:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Selecionar features e alvo
features = ['store_nbr', 'family', 'onpromotion', 'dcoilwtico', 'transactions']
target = 'sales'

# Preencher valores ausentes
imputer = SimpleImputer(strategy='mean')
train[features] = imputer.fit_transform(train[features])
test[features] = imputer.transform(test[features])

# Treinar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[features], train[target])

# Fazer previsões
predictions = model.predict(test[features])

# Salvar resultado no formato de submissão
submission['sales'] = predictions
submission.to_csv('submission.csv', index=False)
