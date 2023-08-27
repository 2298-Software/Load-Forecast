import math

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_yearly

import util
import yaml
from matplotlib import pyplot as plt


def add_seasons_to_df(df):
    df['school_in'] = df['ds'].apply(school_in)
    df['school_out'] = df['ds'].apply(school_out)
    df['spring'] = df['ds'].apply(spring)
    df['summer'] = df['ds'].apply(summer)
    df['fall'] = df['ds'].apply(fall)
    df['winter'] = df['ds'].apply(winter)
    df['spring_break'] = df['ds'].apply(spring_break)
    return df


def apply_configured_seasons(mdl):
    mdl.add_seasonality(name='school_in', period=270, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='school_out', period=120, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='spring', period=120, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='summer', period=120, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='fall', period=120, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='winter', period=120, fourier_order=3, prior_scale=0.1)
    mdl.add_seasonality(name='spring_break', period=7, fourier_order=3, prior_scale=0.1)
    return mdl


def school_in(ds):
    date = pd.to_datetime(ds)
    if date.weekday() and (date.month > 7 or date.month < 5):
        return 1
    else:
        return 0


def school_out(ds):
    date = pd.to_datetime(ds)
    if date.weekday() and (date.month < 7 or date.month > 5):
        return 1
    else:
        return 0


def spring(ds):
    date = pd.to_datetime(ds)
    if date.month in (3, 4, 5):
        return 1
    else:
        return 0


def summer(ds):
    date = pd.to_datetime(ds)
    if date.month in (6, 7, 8):
        return 1
    else:
        return 0


def spring_break(ds):
    date = pd.to_datetime(ds)
    if date.day in (11, 12, 13, 14, 15, 16, 17, 18) and date.month == 3:
        return 1
    else:
        return 0


def fall(ds):
    date = pd.to_datetime(ds)
    if date.month in (9, 10, 11):
        return 1
    else:
        return 0


def winter(ds):
    date = pd.to_datetime(ds)
    if date.month in (12, 1, 2):
        return 1
    else:
        return 0


with open('conf.yml', 'r') as file:
    conf = yaml.safe_load(file)

model_type = 'flat'
history: pd.DataFrame = util.get_voltage_data(conf)
testing_set = conf['model']['testing_set']
row_cnt: int = len(history.index)
print(f"total row cnt is: {row_cnt}")
testing_cnt = math.floor(row_cnt * testing_set)
history = history.head(row_cnt - testing_cnt)

history = add_seasons_to_df(history)

m = Prophet()
m = apply_configured_seasons(m)
m.add_country_holidays(country_name='US')
m.fit(history)

future = m.make_future_dataframe(periods=72, freq='1h', include_history=False)
future = add_seasons_to_df(future)
forecast = m.predict(future)

fig = m.plot_components(forecast)
fig.show()
