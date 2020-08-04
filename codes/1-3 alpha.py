# # -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200731"
#coding=utf-8
import pandas as pd
import numpy as np
from Alpha_generator import Generator
from tqdm import tqdm
import time
def generate_alpha(df):
    g = Generator(df)
    # a1 = g.alpha_001()
    a2 = g.alpha_002()
    a3 = g.alpha_003()
    a4 = g.alpha_004()
    # a5 = g.alpha_005()
    a6 = g.alpha_006()
    # a7 = g.alpha_007()
    # a8 = g.alpha_008()
    # a9 = g.alpha_009()
    a10 = g.alpha_010()
    # a11 = g.alpha_011()
    # a12 = g.alpha_012()
    # a13 = g.alpha_013()
    a14 = g.alpha_014()
    a15 = g.alpha_015()
    # a16 = g.alpha_016()
    # a17 = g.alpha_017()
    a18 = g.alpha_018()
    a19 = g.alpha_019()
    a20 = g.alpha_020()
    # a21 = g.alpha_021()
    a22 = g.alpha_022()
    a23 = g.alpha_023()
    a24 = g.alpha_024()
    # a25 = g.alpha_025()
    # a26 = g.alpha_026()
    a27 = g.alpha_027()
    a28 = g.alpha_028()
    # a29 = g.alpha_029()
    a30 = g.alpha_030()
    a31 = g.alpha_031()
    # a32 = g.alpha_032()
    # a33 = g.alpha_033()
    a34 = g.alpha_034()
    # a35 = g.alpha_035()
    # a36 = g.alpha_036()
    # a37 = g.alpha_037()
    a38 = g.alpha_038()
    # a39 = g.alpha_039()
    # a40 = g.alpha_040()
    # a41 = g.alpha_041()
    # a42 = g.alpha_042()
    # a43 = g.alpha_043()
    # a44 = g.alpha_044()/
    # a45 = g.alpha_045()
    a46 = g.alpha_046()
    a47 = g.alpha_047()
    # a48 = g.alpha_048()
    a49 = g.alpha_049()
    a52 = g.alpha_052()
    a53 = g.alpha_053()
    a54 = g.alpha_054()
    # a56 = g.alpha_056()
    a57 = g.alpha_057()
    a58 = g.alpha_058()
    a59 = g.alpha_059()
    # a60 = g.alpha_060()
    # a61 = g.alpha_061()
    # a62 = g.alpha_062()
    a63 = g.alpha_063()
    # a64 = g.alpha_064()
    a65 = g.alpha_065()
    a66 = g.alpha_066()
    a67 = g.alpha_067()
    # a68 = g.alpha_068()
    # a70 = g.alpha_070()
    a71 = g.alpha_071()
    a72 = g.alpha_072()
    # a74 = g.alpha_074()
    # a76 = g.alpha_076()
    # a77 = g.alpha_077()
    a78 = g.alpha_078()
    a79 = g.alpha_079()
    # a80 = g.alpha_080()
    # a81 = g.alpha_081()
    a82 = g.alpha_082()
    # a83 = g.alpha_083()
    # a84 = g.alpha_084()
    # a85 = g.alpha_085()
    a86 = g.alpha_086()
    # a87 = g.alpha_087()
    a89 = g.alpha_089()
    # a90 = g.alpha_090()
    # a91 = g.alpha_091()
    a93 = g.alpha_093()
    # a94 = g.alpha_094()
    # a95 = g.alpha_095()
    a96 = g.alpha_096()
    # a97 = g.alpha_097()
    a98 = g.alpha_098()
    # a99 = g.alpha_099()
    # a100 = g.alpha_100()
    # a101 = g.alpha_101()
    # a102 = g.alpha_102()
    # a104 = g.alpha_104()
    # a105 = g.alpha_105()
    a106 = g.alpha_106()
    a107 = g.alpha_107()
    # a108 = g.alpha_108()
    a109 = g.alpha_109()
    # a110 = g.alpha_110()
    # a111 = g.alpha_111()
    a112 = g.alpha_112()
    # a113 = g.alpha_113()
    # a114 = g.alpha_114()
    a116 = g.alpha_116()
    # a117 = g.alpha_117()
    a118 = g.alpha_118()
    # a120 = g.alpha_120()
    a122 = g.alpha_122()
    # a123 = g.alpha_123()
    # a124 = g.alpha_124()
    # a125 = g.alpha_125()
    a126 = g.alpha_126()
    a129 = g.alpha_129()
    # a130 = g.alpha_130()
    # a132 = g.alpha_132()
    # a134 = g.alpha_134()
    a135 = g.alpha_135()
    # a136 = g.alpha_136()
    # a139 = g.alpha_139()
    # a141 = g.alpha_141()
    # a142 = g.alpha_142()
    # a144 = g.alpha_144()
    # a145 = g.alpha_145()
    # a148 = g.alpha_148()
    # a150 = g.alpha_150()
    a152 = g.alpha_152()
    a153 = g.alpha_153()
    # a154 = g.alpha_154()
    # a155 = g.alpha_155()
    a158 = g.alpha_158()
    a159 = g.alpha_159()
    a160 = g.alpha_160()
    a161 = g.alpha_161()
    a162 = g.alpha_162()
    # a163 = g.alpha_163()
    a164 = g.alpha_164()
    a167 = g.alpha_167()
    # a168 = g.alpha_168()
    a169 = g.alpha_169()
    # a170 = g.alpha_170()
    a171 = g.alpha_171()
    a172 = g.alpha_172()
    a173 = g.alpha_173()
    a174 = g.alpha_174()
    # a176 = g.alpha_176()
    # a178 = g.alpha_178()
    # a179 = g.alpha_179()
    # a180 = g.alpha_180()
    a184 = g.alpha_184()
    a185 = g.alpha_185()
    a186 = g.alpha_186()
    a187 = g.alpha_187()
    a188 = g.alpha_188()
    a189 = g.alpha_189()
    # a191 = g.alpha_191()

    b = pd.concat(
        [ a2, a3, a4, a6, a10, a14, a15, a18, a19, a20, a22, a23, a24, a28, a31, a34, a38,  a46,a47,
          a49, a52, a53, a54, a57, a58, a59,  a63, a65, a66, a67, a71, a72, a78, a79,  a82, a86, a89,
           a93, a96,a98,  a106, a107, a109, a112, a116,  a118,  a122, a126, a129,  a135,
          a152, a153,a158, a159, a160, a161, a162, a164, a167, a169, a171,a172, a173, a174,
           a184, a185, a186, a187, a188, a189],
        ignore_index=True,
        sort=False,
        axis=1)

    b.columns = ['alpha2', 'alpha3', 'alpha4', 'alpha6', 'alpha10',
                 'alpha14', 'alpha15',  'alpha18', 'alpha19',
                 'alpha20',  'alpha22', 'alpha23', 'alpha24',
                 'alpha28',  'alpha31',  'alpha34', 'alpha38',
                 'alpha46', 'alpha47', 'alpha49', 'alpha52', 'alpha53', 'alpha54',
                 'alpha57', 'alpha58', 'alpha59',  'alpha63', 'alpha65',
                 'alpha66', 'alpha67','alpha71', 'alpha72',
                 'alpha78', 'alpha79', 'alpha82', 'alpha86',  'alpha89',  'alpha93',
                 'alpha96', 'alpha98',  'alpha106', 'alpha107', 'alpha109',
                 'alpha112', 'alpha116',  'alpha118', 'alpha122','alpha126', 'alpha129',
                 'alpha135',  #  data1 = self.close.rolling(21).apply(rolling_div).shift(periods=1)
                 'alpha152', 'alpha153', 'alpha158', 'alpha159', 'alpha160',
                 'alpha161', 'alpha162',  'alpha164', 'alpha167', 'alpha169',
                 'alpha171', 'alpha172', 'alpha173', 'alpha174',
                 'alpha184', 'alpha185', 'alpha186', 'alpha187', 'alpha188', 'alpha189']

    return b

class Para():
    lookback = 300
    pass
para = Para()

if __name__ == '__main__':
    data_original = pd.read_csv('data_original.csv')
    time_list = data_original['Date'].drop_duplicates().to_list()
    for Date in tqdm(time_list[:-1]):
        nextDate = time_list[time_list[Date] + 1]
        data_original.loc[data_original['Date'] == Date, 'ret'] = \
            np.log(np.array(data_original.loc[data_original['Date'] == nextDate, 'Close'])
                   / np.array(data_original.loc[data_original['Date'] == Date, 'Close']))
    Alphas = pd.DataFrame()
    starttime = time.time()
    for currentDate in tqdm(time_list[para.lookback::40]):
        lookbackDate = time_list[time_list[currentDate] - para.lookback]
        dataFrame = data_original.loc[((currentDate - para.lookback) < data_original['Date'])
                                        & (currentDate > data_original['Date']), :]
        alphas = generate_alpha(dataFrame)
        alphas.dropna(axis=0, inplace=True)
        alphas.to_csv('%s'%currentDate+'_alphas.csv')
        # alphas.reset_index() # 把Ticker列剥离
        # alphas.index = [currentDate] * len(alphas.index)
        # Alphas = Alphas.append(alphas)
    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)  # 显示到微秒 程序运行时间：13883.99 s
        # Alphas.append(alphas,ignore_index=True,sort=False)
        
        # data_original.loc[(data_original['Date'] == currentDate)
        #                 ,'alpha1'] = alphas['alpha1']
        # data = pd.concat([data_original.loc[(data_original['Date'] == currentDate)],
        #                   alphas.iloc[-482:,:]],
        #                  # join_axes=[data_original['Ticker']],
        #                  ignore_index=True,
        #                  sort=False)
        # alphas.reset_index()
        # data_original.loc[(data_original['Date'] == currentDate),:].append(alphas.iloc[-482:,:],ignore_index=True,sort=False)
        # data.to_csv('data_191factor.csv')

