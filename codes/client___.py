# # -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200801"
import grpc
import question_pb2, question_pb2_grpc
import contest_pb2, contest_pb2_grpc
import pandas as pd
import numpy as np
import os, time
from universal import tools
from universal import algos
import lightgbm as lgb
from alpha_function import generate_alpha
from sklearn import preprocessing
import logging
import warnings
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
warnings.filterwarnings("ignore")

class main():
    def __init__(self):
        conn = grpc.insecure_channel('101.52.254.180:56701')
        questionClient = question_pb2_grpc.QuestionStub(channel=conn)
        responseQuestion = questionClient.get_question(question_pb2.QuestionResponse(user_id = 22))
        global sequence
        self.sequence = responseQuestion.sequence
        self.capital = responseQuestion.capital
        # self.position = responseQuestion.position
        print('下一次是否可行', responseQuestion.has_next_question)
        print('资本', self.capital)
        print('仓位水平', self.position)
        dailystk = responseQuestion.dailystk
        day_list,code_list,open_list,high_list, low_list, close_list, volume_list= [],[],[],[],[],[],[]
        for i in range(len(dailystk)):
            stock_i = dailystk[i].values
            day_list.append(stock_i[0])
            code_list.append(stock_i[1])
            open_list.append(stock_i[2])
            high_list.append(stock_i[3])
            low_list.append(stock_i[4])
            close_list.append(stock_i[5])
            volume_list.append(stock_i[6])
        global data
        self.data = np.vstack((day_list,code_list,open_list,high_list,low_list,close_list,volume_list)).T
        self.dataFrame = pd.DataFrame(self.data, columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
        pass

    def data_function(self):
        self.data_original = pd.read_csv('CONTEST_DATA_IN_SAMPLE.csv')
        if self.dataFrame['Date'].iloc[-1] < 301:
            self.data_all = self.data_original.append(self.dataFrame)
        else:
            self.data_all = self.data_all.append(self.dataFrame)
        return self.data_all

    def alpha(self):
        alphas = generate_alpha(self.data_all.iloc[-301:-1,:])
        alphas_out = generate_alpha(self.data_all.iloc[-300:, :])
        X_in_sample = np.array(alphas.loc[:, 'alpha_172':'alpha_002'], dtype='<U32')
        y_in_sample = np.array(self.dataFrame.loc[:, 'ret'], dtype='<U32')
        model = lgb.LGBMRegressor(silent=False)
        model.fit(X_in_sample, y_in_sample)
        X_out_of_sample = np.array(alphas_out.loc[:, 'alpha_172':'alpha002'], dtype='<U32')
        y_score_out_of_sample = model.predict(X_out_of_sample)
        self.dataFrame['predict'] = y_score_out_of_sample.copy()
        self.dataFrame['normalized2'] = preprocessing.scale(with_mean=True, X=np.array(self.dataFrame['predict']))
        nor2_des = pd.DataFrame(self.dataFrame['normalized2'].describe()).T
        self.dataFrame.loc[(self.dataFrame['normalized2'] < nor2_des['75%'].iloc[0])
                   & (self.dataFrame['normalized2'] > nor2_des['25%'].iloc[0]), 'normalized2'] = 0
        return

    def position(self,initDf):
        df = pd.concat([pd.Series(initDf['normalized'].to_list()),
                        pd.Series(self.dataFrame['normalized2'].to_list()),
                        pd.Series(self.dataFrame['Close'].to_list()),
                        pd.Series(self.dataFrame['Volume'].to_list())],
                       ignore_index=True,
                       axis=1)
        df.columns = ['normalized', 'normalized2','Close', 'Volume']
        # 限制交易量小于25%分位数的不操作
        volume_des = pd.DataFrame(df['Volume'].describe()).T
        limit25 = volume_des['25%'].iloc[0]
        df.loc[df['Volume'] < limit25, 'Weight'] = 0
        df['Amount1'] = self.capital * df['normalized'] /  df['Close']
        df['Amount1'] = preprocessing.scale(with_mean=True, X=np.array(df['Amount1']))
        df['Amount1'] = df['Amount1'] * 4 / 5
        df['Amount2'] = self.capital * df['normalized2'] / df['Close']
        df['Amount2'] = preprocessing.scale(with_mean=True, X=np.array(df['Amount2']))
        df['Amount2'] = df['Amount2'] * 4 / 5
        self.target_position_pre = (df['Amount1'] + df['Amount2']).to_list()
        self.target_position_pre = self.capital * df['normalized'] / (df['Close'].mean() * df['Close'])

        return self.target_position_pre


    def position_check(self):
        market_value_sup = abs(np.array(self.target_position_pre)) * self.dataFrame['Close']
        target_position_dealt_ = []
        high_position = market_value_sup/ sum(market_value_sup)
        for i in range(len(high_position)):
            if high_position[i] == 0:
                high_position[i] = 0.001
        percent_position = [position /percent *0.01 for percent,position in zip(high_position,self.target_position_pre)]
        limit_trad_vol_position = self.dataFrame['Close'] * self.dataFrame['Volume'] * 100 * 0.05
        over_long_short = np.sum(self.target_position_pre * self.dataFrame['Close'])
        long_short_capital = np.sum(np.abs(self.target_position_pre * self.dataFrame['Close']))
        over_long_short_percent = over_long_short/long_short_capital
        long_short_position = [i/(0.5+0.5*over_long_short_percent)*0.6 if np.sign(over_long_short_percent)==np.sign(i)
                               else i/(0.5-0.5*over_long_short_percent)*0.4 for i in self.target_position_pre]
        for percent,limit_vol,long_short,position in zip(percent_position,limit_trad_vol_position,long_short_position,self.target_position_pre):
            target_position = min( abs(percent),abs(limit_vol),abs(long_short),abs(position))
            target_position_dealt_.append(target_position*np.sign(position))
        print(f'>>单股金额是否超过总占用资金10%测试成功,单只股票最高占用资金比例:{max(high_position)}')
        self.target_position_dealt_ = target_position_dealt_
        return self.target_position_dealt_

    def contestMain(self):
        conn = grpc.insecure_channel('101.52.254.180:56702')
        client = contest_pb2_grpc.ContestStub(channel=conn)
        responseLogin = client.login(contest_pb2.LoginRequest(user_id=22,
                                                                user_pin='gwGCvRgR'))
        print('capital', responseLogin.init_capital)
        responseSubmit = client.submit_answer(contest_pb2.AnswerRequest(
                                                 user_id = 22,
                                                 user_pin = 'gwGCvRgR',
                                                 session_key = responseLogin.session_key,
                                                 sequence = self.sequence,
                                                 positions = self.target_position_dealt_
                                                ))
        print(responseSubmit)
        return


if __name__ == '__main__':
    ############################################################################################################
    # in smaple 的权重是 init.csv
    # 后面新跑的注意修改文件名称
    weight_init = pd.read_csv('init.csv')
    # weight_init = pd.read_csv('init2.csv')
    ############################################################################################################
    starttime = time.time()
    while True:
        Main = main()
        Main.data_function()
        Main.alpha()
        Main.position(weight_init)
        Main.position_check()
        Main.contestMain()
        time.sleep(5)
        endtime = time.time()
        dtime = endtime - starttime
        if dtime / (5 * 72) == int:
            data = Main.data_all()
            ############################################################################################################
            # 跑的慢的话就把数据存下来
            # data.to_csv('data_original2.csv',index_col=0)
            ############################################################################################################
            # 可以跑的话就使用online portfolio selection
            algo = algos.OLMAR(window=5, eps=10)
            result = algo.run(data)
            most_profitable = result.equity_decomposed.iloc[-1]
            result.fee = 0.001
            print(result.summary())
            stock_1st_selection = pd.read_csv('stock_1st_selection.csv', index_col=0)
            weight = pd.concat([stock_1st_selection, most_profitable], axis=1, sort=False)
            weight.columns = ['1st_selection', '2nd_selection']
            weight.loc[weight['1st_selection'] == 0, '3nd_selection'] = 0
            weight.loc[weight['1st_selection'] == 1, '3nd_selection'] = weight.loc[
                weight['1st_selection'] == 1, '2nd_selection']
            Des = pd.DataFrame(weight['3nd_selection'].describe()).T
            mean = Des['mean'].iloc[0]
            weight['3nd_selection'] = weight['3nd_selection'].replace(0, np.nan)
            weight['normalized'] = preprocessing.scale(with_mean=True, X=np.array(weight['3nd_selection']))
            weight['normalized'] = weight['normalized'].replace(np.nan, 0)
            nor_des = pd.DataFrame(weight['normalized'].describe()).T
            # weight.loc[(weight['normalized'] < nor_des['75%'].iloc[0])
            #            & (weight['normalized'] > nor_des['25%'].iloc[0]), 'normalized'] = 0

            ############################################################################################################
            # 在线跑的话注意修改文件名称
            pd.DataFrame(weight['normalized']).to_csv('init2.csv')
            ############################################################################################################
        else:
            pass


