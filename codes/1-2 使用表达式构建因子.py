# -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200730"

# In[]
import numpy as np
import pandas as pd
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

data_original = pd.read_csv('data_original.csv',index_col = 0)
data_original['pct'] = data_original['Close'].pct_change(periods=1)
data = data_original.iloc[:,1:-1].values
target = data_original['pct'].values

# In[]
def _rolling_rank(data):
    value = rankdata(data)[-1]

    return value


def _rolling_prod(data):
    return np.prod(data)


def _ts_sum(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
    value = np.nan_to_num(value)

    return value


def _sma(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
    value = np.nan_to_num(value)

    return value


def _stddev(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)

    return value


def _ts_rank(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_rank).tolist())
    value = np.nan_to_num(value)

    return value


def _product(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_prod).tolist())
    value = np.nan_to_num(value)

    return value


def _ts_min(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)

    return value


def _ts_max(data):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)

    return value


def _delta(data):
    value = np.diff(data.flatten())
    value = np.append(0, value)

    return value


def _delay(data):
    period = 1
    value = pd.Series(data.flatten()).shift(1)
    value = np.nan_to_num(value)

    return value


def _rank(data):
    value = np.array(pd.Series(data.flatten()).rank().tolist())
    value = np.nan_to_num(value)

    return value


def _scale(data):
    k = 1
    data = pd.Series(data.flatten())
    value = data.mul(1).div(np.abs(data).sum())
    value = np.nan_to_num(value)

    return value


def _ts_argmax(data):
    window = 10
    value = pd.Series(data.flatten()).rolling(10).max() + 1
    value = np.nan_to_num(value)

    return value


def _ts_argmin(data):
    window = 10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmin) + 1
    value = np.nan_to_num(value)

    return value

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
# init_function = ['add', 'sub', 'mul', 'div']
# 自定义函数, make_function函数群

# make_function函数群
delta = make_function(function=_delta, name='delta', arity=1)
delay = make_function(function=_delay, name='delay', arity=1)
rank = make_function(function=_rank, name='rank', arity=1)
scale = make_function(function=_scale, name='scale', arity=1)
sma = make_function(function=_sma, name='sma', arity=1)
stddev = make_function(function=_stddev, name='stddev', arity=1)
product = make_function(function=_product, name='product', arity=1)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
ts_min = make_function(function=_ts_min, name='ts_min', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=1)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=1)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=1)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=1)

user_function = [delta, delay, rank, scale, sma, stddev, product, ts_rank, ts_min, ts_max, ts_argmax, ts_argmin, ts_sum]

# In[]
# scale(ts_sum(min(ts_argmin(ts_min(max(High, High))), log(sqrt(ts_argmax(-0.481))))))
# alpha5很可能无效，sqrt(-0.481)
# def alpha_5(df):
#     value1 = ts_argmin(ts_min(np.array(df['High'])))
#     value2 = np.log(np.sqrt(-0.481))
#     value2 = np.array([0] * len(value1))
#     value = []
#     for i in range(len(value1)):
#         value.append(scale(ts_sum(min(value1[i],value2[i]))))
#     # value = scale(ts_sum(min(value1.any(),value2.any())))
#     # value = np.min(scale(ts_sum(np.array(value1))) ,scale(ts_sum(np.array(value2))))
#     # value = scale(ts_sum(min(value1,value2)))
#                              # np.log(np.sqrt(-0.481)))))
#
#     return value[0]

# In[]
# ts_min(ts_sum(div(min(ts_rank(Low), sub(Volume, Volume)), sma(ts_max(Low)))))
def alpha_6(df):
    value11 = list(map(lambda x: x[0]-x[1], zip(df['Volume'].to_list(), df['Volume'].to_list())))
    value1 = []
    for i in range(len(value11)):
        value1.append(min(ts_rank(np.array(df['Low']))[i],value11[i]))
    # value1 = min(ts_rank(np.array(df['Low'])),value11)
    value2 = sma(ts_max(np.array(df['Low'].to_list())))
    value = ts_min(ts_sum(value1+value2))
    return value

# In[]
# inv(stddev(add(inv(min(Close, Open)), inv(product(Volume)))))
def alpha_9(df):
    # value1 = 1 / (min(df['Close'].to_list(), df['Open'].to_list()))
    value1 = list(map(lambda x: x[0] / x[1], zip(df['Close'].to_list(), df['Open'].to_list())))
    value2 = 1 / product(np.array(df['Volume']))
    value = 1 / stddev(value1 + value2)
    return value

# In[]
# tan(log(inv(ts_min(inv(ts_max(Open))))))
def alpha_10(df):
    value = np.tan(np.log(1 / (ts_min(1 / ts_max(np.array(df['Open']))))))
    return value

# In[]
# ts_max(tan(ts_max(sub(Open, Open))))
def alpha_7(df):
    value1 = list(map(lambda x: x[0] - x[1], zip(df['Open'].to_list(), df['Open'].to_list())))
    value = ts_max(np.tan(ts_max(np.array(value1))))
    return value

# In[]
# product(scale(sqrt(inv(scale(Low)))))
def alpha_8(df):
    value = product(scale(np.sqrt(1 / scale(np.array(df['Low'])))))
    return value

# In[]
# log(scale(log(Close)))
def alpha_4(df):
    value = np.log(scale(np.log(df['Close'].to_list())))
    return value

# In[]
# tan(ts_min(inv(Volume)))
def alpha_3(df):
    value = np.tan(ts_min(1/ (np.array(df['Volume']))))
    return value

# In[]
# sqrt(delta(product(-0.461)))
def alpha_2(df):
    value = np.sqrt(-0.461)
    return value

# In[]
# sub(mul(product(product(ts_min(Low))), div(inv(scale(Low)), sin(inv(0.033)))), product(ts_rank(ts_rank(mul(Open, Open)))))
def alpha_1(df):
    value11 = (1 / (scale(np.array(df['Low']))))/ (np.sin(1/0.033))
    value12 = product(product(ts_min(np.array(df['Low']))))
    value1 = value12 * value11
    # value1 = pd.Series(value12).mul(value11)
    value2 = product(ts_rank(df['Open'].to_list() * df['Open'].to_list()))
    value = value1 - value2
    return value
# In[]
# data_original['alpha_5'] = alpha_5(data_original) # False
data_original['alpha_6'] = alpha_6(data_original)##########
# data_original['alpha_1'] = alpha_1(data_original)#########
# data_original['alpha_2'] = alpha_2(data_original) # nan
# data_original['alpha_3'] = alpha_3(data_original) # 0
# data_original['alpha_4'] = alpha_4(data_original) # 没问题


# data_original['alpha_7'] = alpha_7(data_original) # 0
# data_original['alpha_8'] = alpha_8(data_original) # 0
# data_original['alpha_9'] = alpha_9(data_original)# inf
# data_original['alpha_10'] = alpha_10(data_original)# 没问题
# data_original.to_csv('data_factor.csv')