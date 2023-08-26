import math

import pandas as pd
import numpy as np
import yaml

def generate_voltage_data():
    date_range = pd.date_range(start='1/1/2022', end='1/1/2023')
    history = pd.DataFrame(date_range, columns=['ds'])
    history['y'] = np.random.uniform(800, 820, size=(len(date_range)))
    return history


def get_voltage_data(conf):
    data_path = conf['data']['path']
    training_set = conf['model']['training_set']
    history = pd.read_csv(data_path)
    row_cnt = len(history.index)
    training_cnt = math.floor(row_cnt * training_set)
    # eliminate test rows from training set
    history = history.head(training_cnt)
    history['ds'] = history['datetime']
    history['y'] = history['nat_demand']
    history = history[['ds', 'y']]
    limit = conf['data']['limit']
    if limit > 0:
        history = history.head(limit)
    return history
