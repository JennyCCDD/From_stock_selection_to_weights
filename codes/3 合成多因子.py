# # -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200730"

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import performance
import lightgbm as lgb
from sklearn import preprocessing
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data_factor = pd.read_csv('data_original.csv')
# data_factor = pd.read_csv('data_factor.csv',index_col = 0)
data_factor['pct'] = data_factor['Close'].pct_change(periods=1)
# factors = data_factor.loc[:,'alpha_1':'alpha_10'].values
target = data_factor['pct'].values

# DES = pd.DataFrame(factors.describe())
# DES['all'] = DES.apply(lambda x: x.sum(), axis = 1)
class Para():
    groupnum = 5
    lookback = 300
    factor = 'LGBM'
    pass
para = Para()
meanlist = []
def portfolio_test(meanDf):
    sharp_list = []
    ret_list = []
    std_list = []
    mdd_list = []
    compare = pd.DataFrame()
    for oneleg in tqdm(range(len(meanDf.columns))):
        portfolioDF = pd.DataFrame()
        portfolioDF['ret'] = meanDf.iloc[:, oneleg]
        portfolioDF['nav'] = (portfolioDF['ret'] + 1).cumprod()
        performance_df = performance(portfolioDF, para)
        sharp_list.append(np.array(performance_df.iloc[:, 0].T)[0])
        ret_list.append(np.array(performance_df.iloc[:, 1].T)[0])
        std_list.append(np.array(performance_df.iloc[:, 2].T)[0])
        mdd_list.append(np.array(performance_df.iloc[:, 3].T)[0])
        compare[str(oneleg)] = portfolioDF['nav']
    performanceDf = pd.concat([pd.Series(sharp_list),
                               pd.Series(ret_list),
                               pd.Series(std_list),
                               pd.Series(mdd_list)],
                              axis=1, sort=True)
    performanceDf.columns = ['Sharp',
                             'RetYearly',
                             'STD',
                             'MDD']
    compare.index = meanDf.index
    plt.plot(range(len(compare.iloc[1:, 1])),
             compare.iloc[1:, :])
    plt.title(para.factor)
    plt.grid(True)
    plt.legend()
    plt.savefig(para.factor + '_performance_nav.png')
    plt.show()
    return performanceDf, compare
time_list = data_factor['Date'].drop_duplicates().to_list()

# In[]
for currentDate in tqdm(time_list[:-2]):
    nextDate = time_list[time_list[currentDate] + 1]
    data_factor.loc[data_factor['Date'] == currentDate,'ret'] = \
         np.log(np.array(data_factor.loc[data_factor['Date'] == nextDate, 'Close'])
                  / np.array(data_factor.loc[data_factor['Date'] == currentDate, 'Close']))
data_factor.fillna(method = 'ffill',inplace=True)
data_factor.replace(np.inf,0)
data_factor.replace(np.nan,0)
data_factor.replace('#NAME?',0)
ALPHAS = pd.read_csv('all_alphas.csv',index_col=0)
# data_factor = pd.to_numeric(data_factor,errors='coerce')
# In[]
COEF = []
data_factor['predict'] = ''
for currentDate in tqdm(time_list[para.lookback * 2::]):
    lookbackDate = time_list[time_list[currentDate] - para.lookback]
    dataFrame = data_factor.loc[(lookbackDate < data_factor['Date'])&(currentDate  > data_factor['Date']),:]
    dataFrame_out = data_factor.loc[((lookbackDate+1) < data_factor['Date'])&((currentDate+1)  > data_factor['Date']), :]
    # number = preprocessing.LabelEncoder()
    # dataFrame = number.fit_transform(dataFrame)
    alphas = ALPHAS.loc[(lookbackDate < ALPHAS.index)&(currentDate > ALPHAS.index),:]
    alphas_out = ALPHAS.loc[((lookbackDate+1) < ALPHAS.index)&((currentDate+1)> ALPHAS.index), :]
    dataFrame = dataFrame.iloc[-len(alphas.index):,:]
    dataFrame_out = dataFrame_out.iloc[-len(alphas_out.index):, :]
    X_in_sample = np.array(alphas.loc[:, 'alpha2':'alpha189'],dtype='<U32')
    # X_in_sample = dataFrame.loc[:, 'Open':'Volume'].values
    y_in_sample = np.array(dataFrame.loc[:,'ret'],dtype='<U32')
    # dataFrame_out = number.fit_transform(dataFrame_out)
    X_out_of_sample = np.array(alphas_out.loc[:,'alpha2':'alpha189'],dtype='<U32')
    X_out_of_sample = X_out_of_sample.astype('float')
    # X_out_of_sample = dataFrame_out.loc[:, 'Open':'Volume'].values
    y_out_of_sample = np.array(dataFrame_out.loc[:,'ret'],dtype='<U32')
    model = lgb.LGBMRegressor(silent=False)
    model.fit(X_in_sample, y_in_sample)
    y_score_out_of_sample = model.predict(X_out_of_sample)
    coef = model.feature_importances_
    COEF.append(coef)
    dataFrame_out['predict'] = y_score_out_of_sample.copy()

    data_factor.loc[((lookbackDate+1)<data_factor['Date'])&((currentDate+1)> data_factor['Date'])]\
        .iloc[-len(alphas_out.index):,-1] = \
        y_score_out_of_sample.copy()
    dataFrame_out.sort_values(by='predict', ascending=False)
    Des = dataFrame_out['predict'].describe()
    dataFrame_out['Score'] = ''
    eachgroup = int(Des['count'] / para.groupnum)
    for groupi in range(0, para.groupnum - 1):
        dataFrame_out.iloc[groupi * eachgroup:(groupi + 1) * eachgroup, -1] = groupi + 1
    dataFrame_out.iloc[(para.groupnum - 1) * eachgroup:, -1] = para.groupnum
    dataFrame_out['Score'].type = np.str
    meanlist.append(np.array(dataFrame_out.groupby('Score')['ret'].mean()))

# In[]
meanDf = pd.DataFrame(meanlist,index = time_list[para.lookback * 2::])
model_coef = pd.DataFrame(COEF,columns=ALPHAS.columns)
# model_coef = pd.DataFrame(COEF,columns=data_factor.loc[:, 'Open':'Volume'].columns)
model_coef.loc['mean',:] = model_coef.apply(lambda x: x.mean())
portfolio_test(meanDf.dropna())


