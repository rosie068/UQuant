import pandas as pd
import scipy.stats as st
import numpy as np

factor = ""

calendar_frame = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20091201",endDate=u"20171231",field=u"",pandas="1")
calendar_frame = calendar_frame[calendar_frame.isMonthEnd == 1][['calendarDate','isMonthEnd']]
#因为有"XSHG,XSHE"的交易日期，所以需要去重一下，要不然每个交易日遍历两次
calendar_frame = calendar_frame.drop_duplicates()
#print(calendar_frame)

factor_list = []
month_return = []

for tdate in calendar_frame.calendarDate.values:
    in_to_Date = tdate.replace('-', '')  #去掉日期之间的链接'-','2018-06-03'
    #DataAPI.IdxConsGet获取国内外指数的成分构成情况，包括指数成分股名称、成分股代码、入选日期、剔除日期等
    #ticker = 000300 means it is in 沪深300
    hs300_frame = DataAPI.IdxConsGet(secID=u"",ticker=u"000300",isNew=u"",intoDate=in_to_Date ,field=u"", pandas="1")
    tickers = hs300_frame.consTickerSymbol.values
    
    #here we get the raw data factors
    factor_frame = DataAPI.MktStockFactorsOneDayGet(tradeDate=tdate,ticker=tickers,field=["secID",'tradeDate','PB'], pandas="1")
    factor_list.append(factor_frame) #把每一天所有股票的dataframe放在一个list里面
    
    month_return1 = DataAPI.MktEqumGet(secID=u"",ticker=tickers, monthEndDate=in_to_Date, isOpen=u"",field=["secID",'endDate','return'], pandas="1")
    month_return.append(month_return1)

#factor_list is a list of factor data of all the stocks with the MonthEnd Dates
#month_return is the list of returns on all the stocks on the monthEnd dates

tfactor_frame = pd.concat(factor_list)
tfactor_frame['tradeDate'] = tfactor_frame['tradeDate'].apply(lambda x: x.replace("-","")) #去掉交易日期中的'-

#winsorize
series = pd.Series(tfactor_frame['PB'])
trans_series = pd.Series(st.mstats.winsorize(series, limits=[0.05, 0.05])) 
tfactor_frame['PB'] = trans_series

#standardize
mean = tfactor_frame['PB'].mean()
std = tfactor_frame['PB'].std()
tfactor_frame['PB'] = (tfactor_frame['PB'] - mean)/std
#print tfactor_frame

monthly_return_frame = pd.concat(month_return)    
monthly_return_frame['endDate'] = monthly_return_frame['endDate'].apply(lambda x: x.replace("-",""))
#print(monthly_return_frame)

####因子的IC = 因子值和股票下一期收益率的截面关系
#取交易日期
calendar_frame1 = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20091201",endDate=u"20171231",field=u"",pandas="1")
#月末交易日期
calendar_frame1 = calendar_frame1[calendar_frame1.isMonthEnd==1][['calendarDate', 'isMonthEnd']]
calendar_frame1['calendarDate'] = calendar_frame1['calendarDate'].apply(lambda x: x.replace("-",""))
calendar_frame1 = calendar_frame1.drop_duplicates()
#交易日往下移一个月
calendar_frame1['prev_month_end'] = calendar_frame1['calendarDate'].shift(1)
calendar_frame1 = calendar_frame1[['calendarDate', 'prev_month_end']]
calendar_frame1.rename(columns={"calendarDate":"month_end"}, inplace=True)
print calendar_frame1.head()

#把因子与日期merge，再与月度收益率merge
tfactor_frame1 = tfactor_frame.copy(deep=True)
tfactor_frame1.drop_duplicates()
#对齐因子
tfactor_frame1 = tfactor_frame1.merge(calendar_frame1, left_on=['tradeDate'], right_on=['prev_month_end'], how='left')
#把日历dataframe与因子dataframe进行merge，日历dataframe按因子dataframe进行广播，194的长度广播到了114600

#对齐收益率
tfactor_frame1 = tfactor_frame1.merge(monthly_return_frame, left_on=['secID', 'month_end'], right_on=['secID', 'endDate'], how='left')

tfactor_frame1 = tfactor_frame1.dropna() #由于有些股票在下一期不在HS300成分股里面，所以下一期的月收益没有，所以会有空值出现，剔除空值进行IC值计算

#print '对齐剔空后的frame：'
#print(tfactor_frame1)

#计算IC值
def calc_normal_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    return tmp_df['PB'].corr(tmp_df['return'])

# 用scipy的包计算
def calc_normal_ic_2(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Pearson correlation coefficient measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed
    return st.pearsonr(tmp_df['PB'], tmp_df['return'])[0]

# 用scipy的包计算
def calc_normal_ic_2_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Pearson correlation coefficient measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed
    return st.pearsonr(tmp_df['PB'], tmp_df['return'])[1]

# 用scipy的包计算
def calc_rank_ic(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.
    return st.spearmanr(tmp_df['PB'], tmp_df['return'])[0]

def calc_rank_ic_p(df):
    tmp_df = df.copy()
    tmp_df.dropna(inplace=True)
    #The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.
    return st.spearmanr(tmp_df['PB'], tmp_df['return'])[1]

'''
# 计算Normal IC
normal_ic_frame = tfactor_frame1.groupby(['tradeDate']).apply(calc_normal_ic)
print "Normal IC1:"
print normal_ic_frame
'''

#计算Normal IC2, same as Normal IC
normal_ic_frame = (tfactor_frame1.groupby(['tradeDate']).apply(calc_normal_ic_2))
normal_ic_frame_p = (tfactor_frame1.groupby(['tradeDate']).apply(calc_normal_ic_2_p))
normal_ic = pd.DataFrame({'IC':normal_ic_frame,'pVal':normal_ic_frame_p})
#print "Normal IC2:"
#print normal_ic.head(20)

# 计算Rank IC
####by tradeDate
#rank_ic_frame = tfactor_frame1.groupby(['tradeDate']).apply(calc_rank_ic)
#rank_ic_frame_p = tfactor_frame1.groupby(['tradeDate']).apply(calc_rank_ic_p)

#####by secID
rank_ic_frame = tfactor_frame1.groupby(['tradeDate']).apply(calc_rank_ic)
rank_ic_frame_p = tfactor_frame1.groupby(['tradeDate']).apply(calc_rank_ic_p)
rank_ic = pd.DataFrame({'IC':rank_ic_frame, 'pVal': rank_ic_frame_p})
#print(rank_ic.head())

sub_l = len(rank_ic)/10
rank_ic['groups'] = pd.Series(len(rank_ic))
for i in range(10):
    if i == 9:
        rank_ic.iloc[i*sub_l:,2] = i
    else:
        rank_ic.iloc[i*sub_l:(i+1)*sub_l,2] = i
#print(rank_ic)

rank_ic_plot = rank_ic.groupby(by='groups').agg({'IC':sum,'pVal':['max','min']}).reset_index()
rank_ic_plot = rank_ic_plot.drop(['groups'],axis=1)
print "Rank IC:"
#print rank_ic.head(20)
#print len(rank_ic)
rank_ic_plot.plot(kind='bar')

###TODO 分组排序,检查因子单一性,线性
######按照PB大小对股票进行分组，从时间序列的角度观察各组的历史累计收益、信息比率、最大回撤以及胜率等。各组表现的优势组的胜率越高，单调性越强，说明指标的区分能力和选股能力越强。

#1. 按因子值分组
ordered_factor = tfactor_frame1.copy(deep=True)
ordered_factor = ordered_factor.sort(columns=['PB','tradeDate'],ascending=False)
#print ordered_factor

lists = []
ordered_factor['groups'] = pd.Series(len(ordered_factor))
sub_length = len(ordered_factor)/10
for n in range(10):
    if n == 9:
        ordered_factor.iloc[n*sub_length:,7] = n
        temp_list = ordered_factor.iloc[n*sub_length:,:]
    else:
        ordered_factor.iloc[n*sub_length:(n+1)*sub_length,7] = n
        temp_list = ordered_factor.iloc[n*sub_length:(n+1)*sub_length,:]
    lists.append(temp_list)
#print ordered_factor
#print(lists)

grouped_factor = ordered_factor.drop(['secID','month_end','prev_month_end','endDate'],axis=1)
#print(grouped_factor)
grouped_factor = grouped_factor.groupby(by='groups').agg({'return':sum}).reset_index()
print("group by 因子值")
grouped_factor.plot(kind='bar')

grouped_list = []
for i in range(len(lists)):
    temp_group = lists[i].groupby(by='tradeDate').agg({'return':sum}).reset_index()
    grouped_list.append(temp_group)
    temp_group.plot()
#print("group by tradeDate 在每个小组里, they all look the same")

###this is a test
test_ic = rank_ic.copy(deep=True)
#print(test_ic)

test_factor = ordered_factor.copy(deep=True)
test_factor = test_factor.groupby(by='tradeDate').agg({'return':sum,'PB':sum}).reset_index()
#print(test_factor)

#test_merge = test_ic.merge(test_factor, left_on=index, right_on='tradeDate')
test_ic = test_factor.join(test_ic, on='tradeDate')
print(test_ic)
