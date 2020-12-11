import pandas as pd 
import pandas.io.data as web
import matplotlib.pyplot as plt 
import datetime as dt 
import numpy as np 

start = dt.datetime(2016,1,1)
end = dt.date.today()-dt.timedelta(days=1) #this is yesterday

#那平安银行为例
out_data = DataAPI.MktEqudGet(secID = u"000001.XSHE", beginDate = start, endDate = end, pandas="1")
print(start,end)

#openPrice float 今开盘
#highestPrice float 最高价
#lowestPrice float 最低价
#closePrice float 今收盘
#turnoverVol float 成交量
#turnoverValue float 成交金额
#dealAmount int 成交笔数
#turnoverRate float 日换手率
#重新设置index和数据
out_data = pd.DataFrame({
	"openPrice": out_data["openPrice"].values,
	"highestPrice": out_data["highestPrice"].values,
	"lowestPrice": out_data["lowestPrice"].values,
	"closePrice": out_data["closePrice"].values,
	"turnoverVol": out_data["turnoverVol"].values,
	"turnoverValue": out_data["turnoverValue"].values,
	"dealAmount": out_data["dealAmount"].values,
	"turnoverRate": out_data["turnoverRate"].values,
},
	index=out_data["tradeDate"].values
	)

print("##### len:", len(out_data))
figsiz_all(18,4)
#subplots(1,3,figsize=(9,3),sharey=True)plt.figure()
out_data["closePrice"].plot(grid=True, figsize=figsiz_all,title="closePrice")

#####计算今日收盘回归
plt.figure() #新创建一个图表
out_data["return"] = np.log(out_data["closePrice"] / out_data["closePrice"].shift(1))
#print(out_data.head())
out_data["return"].plot(grid=True, figsize = figsiz_all, title = "closePrice return")

#####今日收盘价波动
out_data["25d"] = pd.rolling_mean(out_data["closePrice"], window=25)
out_data["50d"] = pd.rollng_mean(out_data["closePrice"],window=50)
print(out_data.head(n=2))
plt.figure()
out_data[["closePrice","25d","50d"]].plot(grid=True, figsize=figsiz_all, title="25d50d")
#可以做成一个function,输入开始时间,结束时间,return a dataFrame

#用今日收盘价格 / 昨天收盘价格得出一个回归值
out_data["return"] = np.log(out_data["closePrice"] / out_data["closePrice"].shift(1))

#计算移动平均线25日和50日
out_data["25d"] = pd.rolling_mean(out_data["closePrice"], window = 25)
out_data["50d"] = pd.rolling_mean(out_data["closePrice"], window = 50)









