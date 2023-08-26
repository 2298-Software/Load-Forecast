import math

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_yearly

import util
import yaml
from matplotlib import pyplot as plt

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
print(history.tail(testing_cnt).to_string())
history = history.head(row_cnt - testing_cnt)


history['school_in'] = history['ds'].apply(school_in)
history['school_out'] = history['ds'].apply(school_out)
history['spring'] = history['ds'].apply(spring)
history['summer'] = history['ds'].apply(summer)
history['fall'] = history['ds'].apply(fall)
history['winter'] = history['ds'].apply(winter)


m = Prophet()
m.add_seasonality(name='school_in', period=7, fourier_order=3, prior_scale=0.1)
m.add_seasonality(name='spring', period=7, fourier_order=3, prior_scale=0.1)
m.add_seasonality(name='summer', period=7, fourier_order=3, prior_scale=0.1)
m.add_seasonality(name='fall', period=7, fourier_order=3, prior_scale=0.1)
m.add_seasonality(name='winter', period=7, fourier_order=3, prior_scale=0.1)
m.add_country_holidays(country_name='US')
m.fit(history)

future = m.make_future_dataframe(periods=72, freq='1h', include_history=False)
future['school_out'] = future['ds'].apply(school_out)
future['school_in'] = future['ds'].apply(school_in)
future['spring'] = future['ds'].apply(spring)
future['summer'] = future['ds'].apply(summer)
future['fall'] = future['ds'].apply(fall)
future['winter'] = future['ds'].apply(winter)

forecast = m.predict(future)

fig = m.plot_components(forecast)
fig.show()

