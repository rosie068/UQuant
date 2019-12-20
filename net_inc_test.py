##兴业手册单因子测试 测试因子为净利润
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics
import math

#####寻找数据，取用股票每个季度末收益(chgPct)
raw_inc = DataAPI.FdmtISQGet(ticker=u"",secID="",endDate="",beginDate="",beginYear=u"2005",endYear=u"2017", reportType=u"",field=u"ticker,secID,endDate,exchangeCD,NIncome",pandas="1")
raw_inc = raw_inc[raw_inc.exchangeCD=='XSHE'][['ticker','secID','endDate','NIncome']]
raw_inc = raw_inc.sort(columns=['ticker','endDate'],ascending=[True,False])
#print raw_inc.head(100)

return_date = raw_inc.iloc[0:44,2]
#print(return_date)

secIDs = raw_inc.secID
secIDs = secIDs.drop_duplicates()
#print(secIDs)

#q_return = []
q_return = DataAPI.MktEquqGet(beginDate=u"20050101",endDate=u"20171231",secID=u"", ticker=u"",field=u"ticker,secID,exchangeCD,endDate,chg,chgPct",pandas="1")
q_return = q_return[q_return.exchangeCD=='XSHE'][['ticker','secID','endDate','chgPct']]
q_return = q_return.sort(columns=['ticker','endDate'],ascending=[True,False])
#chg = q_return['chg'].shift(1)
chg_pct = q_return['chgPct'].shift(1)
#q_return.chg = chg
q_return.chgPct = chg_pct
q_return = q_return.reset_index()
#print(q_return.head(100))

raw_data = raw_inc.merge(q_return,left_index=True,right_index=True)
raw_data = raw_data.drop(['index','ticker_y','secID_y','endDate_y'],axis=1)
#print(raw_data.head(20))

###处理数据，去极值标准化
#去极值：这里把最小的5%设成5th percentile, 把最大的5%设成95th percentile
series = pd.Series(raw_data['NIncome'])
trans_series = pd.Series(st.mstats.winsorize(series, limits=[0.05, 0.05])) 
raw_data['NIncome'] = trans_series

#因子标准化
inc_mean = raw_data['NIncome'].mean()
inc_std = raw_data['NIncome'].std()
raw_data.NIncome = (raw_data.NIncome-inc_mean)/inc_std

#收益率标准化
return_mean = raw_data['chgPct'].mean()
return_std = raw_data['chgPct'].std()
raw_data.chgPct = (raw_data.chgPct-return_mean)/return_std

###因子中性化
#print raw_data
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
#print(mktVal.head(100))
raw_data = raw_data.merge(mktVal,left_on=['secID_x','endDate_x'],right_on=['secID','tradeDate'])
#print(raw_data.head(10))
raw_data = raw_data.sort(columns=['industryID1','tradeDate'])
raw_data = raw_data.dropna()
#print(raw_data)

###turn industry IDs into dummy groups of 1 to however many
array = raw_data.industryID1.values
array=np.asfarray(array,float)
array = array-1031399
raw_data['industry'] = array
#print(raw_data.head(20))
dummy_ind = pd.get_dummies(raw_data['industry'], prefix='ind')
dummy_ind.columns= ['ind1','ind2','ind3','ind4','ind5','ind6','ind7','ind8','ind9','ind10']
#print(dummy_ind)

raw_data = raw_data.join(dummy_ind)
print(raw_data.head())

######回归找残余,然后新的残余就是中性化后的因子值
X=raw_data.loc[:,('marketValue','ind1','ind2','ind3','ind4','ind5','ind6','ind7','ind8','ind9','ind10')]
y=raw_data.loc[:,'NIncome']
linreg = LinearRegression()
model=linreg.fit(X, y)
#print (model)
# 训练后模型截距
intercept = linreg.intercept_
print(intercept)
# 训练后模型权重（特征个数无变化）
coef_arr = linreg.coef_

factor_attr = y - np.matmul(X,(coef_arr.T))
#print(new_NInc)

old_Inc = raw_data.NIncome.values
new_Inc = old_Inc - factor_attr + intercept
raw_data.NIncome = new_Inc
print(raw_data.head())

######找IC，用Pearson找IC,用Spearman找RankIC
# 用scipy的包计算
def calc_normal_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Pearson correlation coefficient measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed
    return st.pearsonr(tmp_df['NIncome'], tmp_df['chgPct'])[0]

def calc_normal_ic_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.pearsonr(tmp_df['NIncome'], tmp_df['chgPct'])[1]

def calc_rank_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.
    return st.spearmanr(tmp_df['NIncome'], tmp_df['chgPct'])[0]

def calc_rank_ic_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return st.spearmanr(tmp_df['NIncome'], tmp_df['chgPct'])[1]

ic_f = raw_data.groupby(['endDate_x']).apply(calc_normal_ic)
ic_p_f = raw_data.groupby(['endDate_x']).apply(calc_normal_ic_p)
normal_ic = pd.DataFrame({'IC':ic_f, 'pVal': ic_p_f})
normal_ic['moving'] = normal_ic.rolling(10)['IC'].mean()

rank_ic_f = raw_data.groupby(['endDate_x']).apply(calc_rank_ic)
rank_ic_p_f = raw_data.groupby(['endDate_x']).apply(calc_rank_ic_p)
rank_ic = pd.DataFrame({'IC':rank_ic_f, 'pVal': rank_ic_p_f})

print("Normal IC is ")
print(normal_ic.head(20))
print("----------------------------------------------")
print("Rank IC is ")
print(rank_ic.head(20))

######寻找个IC指标，帮助分析
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

####结论：IC_IR不高，t统计量也不理想，因子无显著性

#####plot the normal ic
ic_to_plot = normal_ic[['IC']]
print("Normal IC is ")
ic_to_plot.plot(kind='bar')
####plot the rank ic
print("Rank IC is ")
rank_ic_f.plot(kind='bar')

###查看normal IC是否是正态分布
import matplotlib.pyplot as plt
mean1 = ic_f.mean()
std1 = ic_f.std()

def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

# 设定 x 轴前两个数字是 X 轴的开始和结束，第三个数字表示步长，或者区间的间隔长度
x = np.arange(-0.06,0.06,0.02) 
#设定 y 轴，载入刚才的正态分布函数
y = normfun(x, mean1, std1)
plt.plot(x,y)
#画出直方图，最后的“normed”参数，是赋范的意思，数学概念
plt.hist(ic_f, bins=10, rwidth=0.9, normed=True)
plt.title('distribution')
#输出
plt.show()
#############结论：与正态分布还存在一些差异

######查看Normal　IC的相关系数。因子的自相关系数越高，换手率就越低，我们喜欢换手率低的因子
def calc_corr(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return tmp_df['NIncome_now'].corr(tmp_df['NIncome_later'])

factorAuto = pd.DataFrame()
factorAuto['date'] = raw_data['endDate_x']
factorAuto['NIncome_now'] = raw_data['NIncome']
factorAuto['NIncome_later'] = raw_data['NIncome'].shift(1)
FactorAutoCorrelation = factorAuto.groupby('date').apply(calc_corr)
###2017-12-31的下一个季度没有数，我们删掉
#FactorAutoCorrelation=FactorAutoCorrelation.drop(['2017-12-31'])
#print(FactorAutoCorrelation)
FactorAutoCorrelation.plot()

#############################结论：自关联系数中偏高，换手率较低 :)

####按因子值大小将股票分成十组,取所有季度的均值并作对比
ordered_factor = raw_data.copy(deep=True)
ordered_factor = ordered_factor.sort(columns=['NIncome','endDate_x'],ascending=False)
#print ordered_factor

lists = []
ordered_factor['groups'] = pd.Series(len(ordered_factor))
sub_length = len(ordered_factor)/10
for n in range(10):
    if n == 9:
        ordered_factor.iloc[n*sub_length:,5] = n
        temp_list = ordered_factor.iloc[n*sub_length:,:]
    else:
        ordered_factor.iloc[n*sub_length:(n+1)*sub_length,5] = n
        temp_list = ordered_factor.iloc[n*sub_length:(n+1)*sub_length,:]
    lists.append(temp_list)
temp = ordered_factor.copy(deep=True)
temp = temp.sort(['secID_x','endDate_x'], ascending=[True,False])
temp = temp.reset_index()
date_list = temp.endDate_x.iloc[0:44]
#print date_list

big_list = pd.DataFrame()
big_list['date'] = date_list
big_list['G0'] = pd.Series()
big_list['G1'] = pd.Series()
big_list['G2'] = pd.Series()
big_list['G3'] = pd.Series()
big_list['G4'] = pd.Series()
big_list['G5'] = pd.Series()
big_list['G6'] = pd.Series()
big_list['G7'] = pd.Series()
big_list['G8'] = pd.Series()
big_list['G9'] = pd.Series()
big_list = big_list.set_index(['date'])
#print(big_list)

for i in range(len(lists)):
    for dates in date_list:
        templ = lists[i][lists[i].endDate_x==dates]
        string = 'G'+str(i)
        big_list.loc[dates,string] = templ['NIncome'].sum()
#print(big_list)

group_avg = pd.DataFrame()
group_avg['avg'] = pd.Series(np.arange(9))
for i in range(9):
    group_avg.iloc[i,0] = big_list.iloc[:,i].mean()

group_avg.plot(kind='bar')
########结论，单调性较好, 可以进一步做行业分组看一下是否有所不同
