# -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200729"
# revised from https://www.joinquant.com/view/community/detail/6e594923d168b1592e8737c88988d91e?type=1

# In[]
import pandas as pd
import numpy as np
import graphviz
from tqdm import tqdm
import pylab
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import pickle
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
# 系统自带的函数群

"""
Available individual functions are:
‘add’ : addition, arity=2.
‘sub’ : subtraction, arity=2.
‘mul’ : multiplication, arity=2.
‘div’ : protected division where a denominator near-zero returns 1., arity=2.
‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
‘abs’ : absolute value, arity=1.
‘neg’ : negative, arity=1.
‘inv’ : protected inverse where a near-zero argument returns 0., arity=1. 
‘max’ : maximum, arity=2.
‘min’ : minimum, arity=2.
‘sin’ : sine (radians), arity=1.
‘cos’ : cosine (radians), arity=1.
‘tan’ : tangent (radians), arity=1.
"""



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
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmax) + 1
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

def _my_metric(y, y_pred, w):
    value = np.sum(np.abs(y) + np.abs(y_pred))

    return value

my_metric = make_fitness(function=_my_metric, greater_is_better=True)



data_original = pd.read_csv('data_original.csv',index_col = 0)
data_original = data_original.loc[data_original['if'] == 1,:].copy()
data_original.drop(['if'],axis = 1,inplace = True)
data_original['pct'] = data_original['Close'].pct_change(periods=1)
data_original.dropna(axis = 0,inplace=True)
data_original = data_original.groupby('Date').mean()
# In[]
data = data_original.iloc[:,1:-1].values
# In[]
target = data_original['pct'].values

# test_size=0.2
# test_num = int(len(data)*test_size)
# X_train = data[:-test_num]
# X_test = data[-test_num:]
# y_train = np.nan_to_num(target[:-test_num])
# y_test = np.nan_to_num(target[-test_num:])
X_train,X_test, y_train, y_test =train_test_split(data,target,test_size=0.2, random_state=0)
print("finished!")

generations = 5
function_set = init_function + user_function
metric = my_metric
population_size = 100
random_state=0
est_gp = SymbolicTransformer(
                            feature_names=data_original.iloc[:,1:-1].columns,
                            function_set=function_set,
                            generations=generations,
                            metric=metric,
                            population_size=population_size,
                            tournament_size=20,
                            random_state=random_state,
                         )

est_gp.fit(X_train, y_train)

# 将模型保存到本地
with open('gp_model.pkl', 'wb') as f:
    pickle.dump(est_gp, f)

# 获取较优的表达式

best_programs = est_gp._best_programs
best_programs_dict = {}
for p in tqdm(best_programs):
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                       'length': p.length_}

best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
print(best_programs_dict)
best_programs_dict.to_csv('best_programs_dict.csv')

def alpha_factor_graph(num):
    # 打印指定num的表达式图
    factor = best_programs[num - 1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('images/'+'%s'%num, format='png', cleanup=True)
    # pylab.savefig('%s'%num+'.png')
    # plt.savefig('%s'%num+'.png')
    return graph

for i in range(1,11,1):
    graph = alpha_factor_graph(i)


