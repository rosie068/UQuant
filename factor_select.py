import numpy as np 
import pandas as pd 
import datetime as dt 
from pandas import Series, DataFrame, isnull
from datetime import timedelta, datetime
from CAL.PyCAL import *

pd.set_option('display.width', 200)

#定义股票池和计算时所要的时间区间参数. 我们以HS300作为股票池,取近一年的数据
universe = set_universe('HS300')
today = Date.todaysDate()
start_date = (today - Period('1Y')).toDateTime().strftime('%Y%m%d')
end_date = today.toDateTime().strftime('%Y%m%d')
print('start_date ')
print(start_date)
print('end_date ')
print(end_date)

#we want PB price to book value. Price可以通过股票日行情数据的总市值得到
#Book value可以通过财产负债报表的归属母公司所有者权益合计得到
#优矿提供访问股票日行情和资产负债表的API,可以获得数据. 但是在获取财务表时,只能指定一种类型的(季度,半年,年报)
#选择一个类型,做循环查询即可. 可以使用concat函数
market_capital = DataAPI.MktEqudGet(secID = universe, field = ['secID', 'tradeDate', 'marketValue','negMarketValue'],beginDate = start_date, endDate = end_date, pandas = '1')

equity = DataFrame()
for rpt_type in ['Q1','S1','Q3','A']:
	try:
		tmp = DataAPI.FdmtBSGet(secID=universe, field = ['secID','endDate','publishDate','TEquityAttrP'],beginDate = start_date, publishDateEnd = end_date, reportType=rpt_type)
	except:
		tmp = DataFrame()
	equity = pd.concat([equity,tmp],axis = 0)

print('Data of TEquityAttrP: ')
print (equity.head())
print('Data of marketValue: ')
print(market_capital.head())

#我们多取了市值数据,只要最新的即可. 所以我们将数据按股票代码和交易日进行排序, 并按股票代码丢弃重复数据.
#这里我们按照股票代码和交易日金星升序排序, 并丢弃重复值时保留最后一个
market_capital = market_capital.sort(columns = ['secID','tradeDate'], ascending = [True, True])
market_capital = market_capital.drop_duplicates(subset='secID', take_last = True)

#接下来我们查看缺失数据并丢弃
#这里isnull returns a ist of booleans. 若数据确实则为True. 我们在总市值缺失的情况下,若流通市值有数据,则取而代之
#否则两者都缺失的情况下, 丢弃数据
market_capital['marketValue'][isnull(market_capital['marketValue'])] = market_capital['negMarketValue'][isnull(market_capital['marketValue'])]

#然后我们drop社区流通市值这里列, 丢弃缺失值,并且把marketValue 重命名为 numerator
market_capital = market_capital.drop('negMarketValue', axis = 1)
numerator = market_capital.dropna()
numerator.rename(columns={'marketValue':'numerator'}, inplace = True)

#这里是处理好的分子
print(numerator)

#现在我们处理分母数据. 我们也丢弃重复数据
equity = equity.sort(columns=['secID','endDate','publishDate'], ascending = [True, False, False])
equity = equity.dropna()
equity = equity.drop_duplicates(subset='secID')
denominator = equity
denominator.rename(columns={"TEquityAttrP":"denominator"}, inplace = True)


#这是处理好的分母
print(denominator)

#现在我们把两个DataFrame合并, 使用参数 how = 'inner' 保持两者均有的股票
dat_info = numerator.merge(denominator, on='secID', how='inner')

#分母!=0, 我们用很小的书来过滤不合格的数据, 然后添加一列PB
dat_info = dat_info[abs(dat_info['denominator']) >= 1e^-8]
dat_info['PB'] = dat_info['numerator']/dat_info['denominator']

#把股票代码和PB两列去除,把DataFrame变成一个Series
pb_signal = dat_info[['secID','PB']]
pb_signal = pb_signal.set_index('secID')['PB']
print(pb_signal)






#现在我们把以上因子计算过程变成一个函数, 是的可以计算回测开始到结束时间的PB,方便通联的多因子信号分析工具RDP的测试
def str2date(date_str):
	date_obj = dt.datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
	return Date.fromDateTime(date_obj)

def signal_pb_calc(universe, current_date):
	today = str2date(current_date)
	start_date = (today - Period('1Y')).toDateTime().strftime('%Y%m%d')
	end_date = today.toDateTime().strftime('%Y%m%d')
	
	#dealing with numerator
	market_capital = DataAPI.MktEqudGet(secID=universe, field = ['secID','tradeDate','marketValue','negMarketValue','turnoverVol'], beginDate = start_date, endDate = end_date, pandas = '1')
	market_capital- market_capital[market_capital['turnoverVol']>0]
	market_capital = market_capital.sort(columns=['secID','tradeDate'], ascending=[True,True])
	market_capital = market_capital.drop_duplicates(subset='secID', take_last= True)
	market_capital['marketValue'][isnull(market_capital['marketValue'])] = market_capital['negMarketValue'][isnull(market_capital['marketValue'])]
	market_capital = market_capital.drop('negMarketValue', axis = 1)
	numerator = market_capital.dropna()
	numerator.rename(columns={'marketValue':'numerator'}, inplace = True)
	
	#dealing with denominator
	equity = DataFrame()
	for rpt_type in ['Q1','S1','Q3','A']:
		try:
			tmp = DataAPI.FdmtBSGet(secID=universe, field = ['secID','endDate','publishDate','TEquityAttrP'],beginDate = start_date, publishDateEnd = end_date, reportType=rpt_type)
		except:
			tmp = DataFrame()
		equity = pd.concat([equity,tmp],axis = 0)

	equity = equity.sort(columns=['secID','endDate','publishDate'], ascending = [True, False, False])
	equity = equity.dropna()
	equity = equity.drop_duplicates(subset='secID')
	denominator = equity
	denominator.rename(columns={"TEquityAttrP":"denominator"}, inplace = True)

	#merge to calc PB
	dat_info = numerator.merge(denominator, on='secID', how='inner')
	dat_info = dat_info[abs(dat_info['denominator']) >= 1e^-8]
	dat_info['PB'] = dat_info['numerator']/dat_info['denominator']
	pb_signal = dat_info[['secID','PB']]
	pb_signal["secID"]=pb_signal["secID"].apply(lamdba x:x[:6])
	
	return pb_signal




#####下面的代码计算沪深300成分股在一段时间内的PB为信号, 把这些PB数据按照天存为csv文件,再把csv打包成zip
start = datetime(2015,1,1)
end = datetime(2015,4,23)
univ = set_universe('HS300')
cal = Calendar('China.SSE')

all_files = []
today = start
while((today-end).days<0):
	today_CAL = Date.fromDateTime(today)
	if(cal.isBizDay(today_CAL)):
		today_str = today.strftime("%Y%m%d")
		print("Calculating PB values on" + today_str)
		pb_value = signal_pb_calc(univ, today_str)
		file_name = today_str + '.csv'
		pb_value.to_csv(file_name, index=False, header = False)
		all_files.append(file_name)
	today=today+timedelta(days=1)

#exporting all *.csv to PB.zip
zip_files("PB" + "_" start.strftime("%Y%m%d") + "_" end.strftime("%Y%m%d"), all_files)

#delete all .csv files
delete_files(all_files)












