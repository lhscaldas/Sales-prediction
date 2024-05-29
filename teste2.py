from datetime import date, datetime, timedelta
import pandas as pd


def date_split(df, split_date):
        split_datetime = datetime.strptime(split_date, "%Y-%m-%d")
        split_datetime2 = split_datetime + timedelta(days=1)
        split_date2 = split_datetime2.strftime("%Y-%m-%d")
        df_train, df_valid = df.loc[:split_date], df.loc[split_date2:]
        return df_train, df_valid

y = pd.read_csv("train.csv")
y = y.set_index("date")
y = y["sales"]
print(y.head())
# y_train, y_valid = date_split(y,"2017-07-01")