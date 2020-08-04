import pandas as pd
import numpy as np
from tqdm import tqdm
DATA = []
data_original = pd.read_csv('data_original.csv')
time_list = data_original['Date'].drop_duplicates().to_list()
class Para():
    lookback = 300
    pass
para = Para()
for i,currentDate in enumerate(tqdm(time_list[para.lookback::40])):
    if i == 0 or (type((i+1) / 40) == int):
        data = pd.read_csv('%s'%currentDate+'_alphas.csv',index_col=0)
    else:
        data = data.copy()
    data.index = [currentDate] * len(data.index)
    DATA.append(data)
alldata = pd.concat(DATA) #,ignore_index = True)
alldata01 = pd.DataFrame(data = alldata)
alldata01.to_csv('all_alphas_40.csv')