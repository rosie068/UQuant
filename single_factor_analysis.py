##兴业手册单因子测试
##author: 何缘
##2019-10-27
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics
import math

##设定研究因子
factor = 'NIncome'
sDate = "20050101"
eDate = "20171231"

##寻找数据，取用股票每个季度末收益(chgPct)
raw_data = DataAPI.FdmtISQGet(ticker=u"",secID="",endDate="",beginDate="",beginYear=u"2005",endYear=u"2017", reportType=u"",field=['ticker','secID','endDate','exchangeCD',factor],pandas="1")
raw_data = raw_data[raw_data.exchangeCD=='XSHE'][['ticker','secID','endDate',factor]]
raw_data = raw_data.sort(columns=['ticker','endDate'],ascending=[True,False])
#print raw_data.head(100)

date_length = len(raw_data[raw_data.ticker=='000001'])
return_date = raw_data.iloc[0:date_length,2]
#print(return_date)

secIDs = raw_data.secID
secIDs = secIDs.drop_duplicates()
#print(secIDs)

q_return = DataAPI.MktEquqGet(beginDate=u"20050101",endDate=u"20171231",secID=u"", ticker=u"", 
                              field=['ticker','secID','exchangeCD','endDate','chgPct'],pandas="1")
q_return = q_return[q_return.exchangeCD=='XSHE'][['ticker','secID','endDate','chgPct']]
q_return = q_return.sort(columns=['ticker','endDate'],ascending=[True,False])
chg_pct = q_return['chgPct'].shift(1)
q_return.chgPct = chg_pct
q_return = q_return.reset_index()
#print(q_return.head(100))

raw_data = raw_data.merge(q_return,left_index=True,right_index=True)
raw_data = raw_data.drop(['index','ticker_y','secID_y','endDate_y'],axis=1)
#print(raw_data.head(20))

##处理数据，去极值标准化中性化
'''
#去极值：这里把最小的5%设成5th percentile, 把最大的5%设成95th percentile
series = pd.Series(raw_data[factor])
trans_series = pd.Series(st.mstats.winsorize(series, limits=[0.05, 0.05])) 
raw_data[factor] = trans_series
'''
#MAD去极值
series = pd.Series(raw_data[factor])
median = series.median()
series = abs(series - median)
MAD_j = series.median()
MAD_ej = 1.483*MAD_j
series[series<median-3*MAD_ej] = median-3*MAD_ej
series[series>median+3*MAD_ej] = median+3*MAD_ej
raw_data[factor] = series

#因子标准化
fac_mean = raw_data[factor].mean()
fac_std = raw_data[factor].std()
raw_data[factor] = (raw_data[factor]-fac_mean)/fac_std

#收益率标准化
return_mean = raw_data['chgPct'].mean()
return_std = raw_data['chgPct'].std()
raw_data.chgPct = (raw_data.chgPct-return_mean)/return_std

##因子中性化
mktList = []
for dates in return_date.values:
    temp_val = DataAPI.MktEqudGet(secID=u"",ticker=u"",tradeDate=dates,beginDate=u"",
        endDate=u"",isOpen="",field=u"ticker,secID,exchangeCD,tradeDate,marketValue",pandas="1")
    mktList.append(temp_val)
mktVal = pd.concat(mktList)
mktVal = mktVal[mktVal.exchangeCD=='XSHE'][['ticker','secID','tradeDate','marketValue']]
mktVal = mktVal.sort(columns=['ticker','tradeDate'],ascending=[True,False])
mktVal = mktVal.drop(['ticker'],axis=1)
#print(mktVal.head(20))

indus = []
for tic in secIDs.values:
    temp_ind = DataAPI.EquIndustryGet(secID=tic,ticker=u"",industryVersionCD=u"",industry=u"中证行业分类（2016版）",industryID=u"",industryID1=u"",industryID2=u"",industryID3=u"",intoDate=u"", field=u"ticker,secID,exchangeCD,industryID1,industryName1",pandas="1")
    indus.append(temp_ind)
industry = pd.concat(indus)
industry = industry[industry.exchangeCD=='XSHE'][['ticker','secID','industryID1','industryName1']]
industry = industry.sort(columns=['ticker'])
industry = industry.set_index('secID')
industry = industry.drop(['ticker'],axis=1)
#print(industry.head(20))

mktVal = mktVal.join(industry,on='secID')
raw_data = raw_data.merge(mktVal,left_on=['secID_x','endDate_x'],right_on=['secID','tradeDate'])
raw_data = raw_data.sort(columns=['industryID1','tradeDate'])
raw_data = raw_data.dropna()
#print(raw_data.head(100))

##turn industry IDs into dummy groups of 1 to however many
array = raw_data.industryID1.values
array=np.asfarray(array,float)
array = array-1031399
raw_data['industry'] = array
dummy_ind = pd.get_dummies(raw_data['industry'], prefix='ind')
dummy_ind.columns= ['ind1','ind2','ind3','ind4','ind5','ind6','ind7','ind8','ind9','ind10']
#print(dummy_ind)

raw_data = raw_data.join(dummy_ind)
#print(raw_data.head(20))

##回归找残余,然后新的残余就是中性化后的因子值
X=raw_data.loc[:,('marketValue','ind1','ind2','ind3','ind4','ind5','ind6','ind7','ind8','ind9','ind10')]
y=raw_data.loc[:,'NIncome']
linreg = LinearRegression()
model=linreg.fit(X, y)
print (model)
intercept = linreg.intercept_
coef_arr = linreg.coef_
factor_attr = y - np.matmul(X,(coef_arr.T))
#print(new_NInc)

old_Inc = raw_data.NIncome.values
new_Inc = old_Inc - factor_attr + intercept
raw_data.NIncome = new_Inc
#print(raw_data.head())

##找IC，用Pearson找IC,用Spearman找RankIC
# 用scipy的包计算
#The Pearson correlation coefficient measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed
def calc_normal_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.pearsonr(tmp_df[factor], tmp_df['chgPct'])[0]

def calc_normal_ic_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.pearsonr(tmp_df[factor], tmp_df['chgPct'])[1]

#The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.
def calc_rank_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.spearmanr(tmp_df[factor], tmp_df['chgPct'])[0]

def calc_rank_ic_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.spearmanr(tmp_df[factor], tmp_df['chgPct'])[1]

ic_f = raw_data.groupby(['endDate_x']).apply(calc_normal_ic)
ic_p_f = raw_data.groupby(['endDate_x']).apply(calc_normal_ic_p)
normal_ic = pd.DataFrame({'IC':ic_f, 'pVal': ic_p_f})
normal_ic['moving'] = normal_ic.rolling(20)['IC'].mean()

rank_ic_f = raw_data.groupby(['endDate_x']).apply(calc_rank_ic)
rank_ic_p_f = raw_data.groupby(['endDate_x']).apply(calc_rank_ic_p)
rank_ic = pd.DataFrame({'IC':rank_ic_f, 'pVal': rank_ic_p_f})

'''
print("Normal IC is ")
print(normal_ic.head(20))
print("----------------------------------------------")
print("Rank IC is ")
print(rank_ic.head(20))
'''

##寻找个IC指标，帮助分析
#1. 历史IC均值及标准差
ic_mean = normal_ic['IC'].mean()
ic_std = normal_ic['IC'].std()
print("IC 均值： " + str(ic_mean*100) + "%")
print("IC 标准差： " + str(ic_std*100) + "%")

#2. 历史IC最大和最小值
ic_max = normal_ic['IC'].max()
ic_min = normal_ic['IC'].min()
print("IC 最大值： " + str(ic_max*100) + "%")
print("IC 最小值： " + str(ic_min*100) + "%")

#3. 收益风险比IC-IR，我们喜欢越高越好
ic_ir = ic_mean/ic_std
print("IC_IR: " + str(ic_ir))

#4. IC的t统计量，用来测量因子的显著性，通常绝对值>2
ic_t = ic_ir * math.sqrt(len(return_date)-2)
print("IC的t统计量: " + str(ic_t))
