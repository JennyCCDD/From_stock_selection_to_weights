# # -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200730"

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import performance
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_factor = pd.read_csv('data_factor.csv',index_col = 0)
data_factor['pct'] = data_factor['Close'].pct_change(periods=1)
factors = data_factor.loc[:,'alpha_1':'alpha_10'].values
target = data_factor['pct'].values

DES = pd.DataFrame(factors.describe())
DES['all'] = DES.apply(lambda x: x.sum(), axis = 1)
class Para():
    groupnum = 5
    factor = 'alpha_1'
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
for currentDate in tqdm(data_factor['Date'][1:-1]):
    nextDate = data_factor['Date'][1:-1][data_factor['Date'][1:-1].index(currentDate) + 1]
    ret = np.log(data_factor['Close'].loc[nextDate, :] / data_factor['Close'].loc[currentDate, :])
    dataFrame = data_factor.loc[data_factor['Date'] == currentDate,:]
    dataFrame.sort_values(by = para.factor, ascending = False)
    Des = dataFrame['factor'].describe()
    dataFrame['Score'] = ''
    eachgroup = int(Des['count'] / para.groupnum)
    for groupi in range(0, para.groupnum - 1):
        dataFrame.iloc[groupi * eachgroup:(groupi + 1) * eachgroup, -1] = groupi + 1
    dataFrame.iloc[(para.groupnum - 1) * eachgroup:, -1] = para.groupnum
    dataFrame['Score'].type = np.str
    meanlist.append(np.array(dataFrame.groupby('Score')['RET'].mean()))
meanDf = pd.DataFrame(meanlist,index = data_factor['Date'][1:-1])
portfolio_test(meanDf)




