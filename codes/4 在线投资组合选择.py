# # -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200731"
# %matplotlib inline
import pandas as pd
import numpy as np
from universal import tools
from universal import algos
from sklearn import preprocessing
import logging
import matplotlib
import time
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
matplotlib.rcParams['savefig.dpi'] = 1.5
# In[]
############################################################################################################
# 后面几次
data_original = pd.read_csv('data_original.csv')
# data_original = pd.read_csv('data_original2.csv')
############################################################################################################
data = pd.Series(list(data_original['Close']), index=[data_original['Date'], data_original['Ticker']]).unstack()

# In[]
starttime = time.time()
algo = algos.OLMAR(window=5, eps=10)
result = algo.run(data)
most_profitable = result.equity_decomposed.iloc[-1]
result.fee = 0.001
print(result.summary())
# result.plot(weights=False, assets=False, ucrp=True, logy=True)
endtime = time.time()
dtime = endtime - starttime
print("程序运行时间：%.8s s" % dtime)  #显示到微秒
# In[]
stock_1st_selection = pd.read_csv('stock_1st_selection.csv',index_col=0)
weight = pd.concat([stock_1st_selection,most_profitable],axis=1,sort=False)
weight.columns = ['1st_selection','2nd_selection']
weight.loc[weight['1st_selection'] == 0,'3nd_selection'] =0
weight.loc[weight['1st_selection'] == 1,'3nd_selection'] =weight.loc[weight['1st_selection'] == 1,'2nd_selection']
Des = pd.DataFrame(weight['3nd_selection'].describe()).T
mean = Des['mean'].iloc[0]
weight['3nd_selection'] = weight['3nd_selection'].replace(0,np.nan)
weight['normalized'] = preprocessing.scale(with_mean = True,X=np.array(weight['3nd_selection']))
weight['normalized'] = weight['normalized'].replace(np.nan,0)
nor_des = pd.DataFrame(weight['normalized'].describe()).T
# weight.loc[(weight['normalized'] < nor_des['75%'].iloc[0])
#             & (weight['normalized'] > nor_des['25%'].iloc[0]),'normalized'] = 0

############################################################################################################
# 后面几次
pd.DataFrame(weight['normalized']).to_csv('init.csv')
# pd.DataFrame(weight['normalized']).to_csv('init2.csv')
############################################################################################################

# In[]
# weight['close'] = data_original['Close'].iloc[-482:].to_list()
# capital = 1000
# weight['amount'] = capital*weight['normalized']/weight['close']
# for i in range(10):
#     df = df.append(data_original.iloc[-482:, :])
# df = pd.concat([pd.Series(weight['normalized'].to_list()),
#                 pd.Series(data_original['Close'].iloc[-482:].to_list()),
#                 pd.Series(data_original['Volume'].iloc[-482:].to_list())],
#                ignore_index=True,
#                axis=1)
# df.columns = ['weight', 'close', 'volume']
# volume_des = pd.DataFrame(df['volume'].describe()).T
# limit25 = volume_des['25%'].iloc[0]
# df.loc[df['volume'] < limit25, 'weight'] = 0
# df['amount'] = capital * df['weight'] / df['close']
# target_position_pre = df['amount'].to_list()

# weight['normalized'] = weight['3nd_selection']/weight['3nd_selection'].sum()
# weight['normalized'] = preprocessing.MaxAbsScaler().fit_transform(np.array(weight['3nd_selection']).reshape(-1, 1))
#.reshape(-1, 1)).fit_transform()
# weight.loc[weight['3nd_selection']>= mean,'normalized'] = \
#     weight.loc[weight['3nd_selection']>=mean,'3nd_selection']/ ((1-mean) * weight.loc[weight['3nd_selection']>=mean,'3nd_selection'].sum())
# weight.loc[weight['3nd_selection']<mean,'normalized'] = \
#     weight.loc[weight['3nd_selection']<mean,'3nd_selection']/ (-1+mean) # weight.loc[weight['3nd_selection']<mean,'3nd_selection'].sum()
# #
