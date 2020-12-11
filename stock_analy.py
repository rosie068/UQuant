import math
import numpy as np 
import pandas as pd 
import tushare as ts 
import datetime
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

begin_time = '2017-02-01'
end_time = '2017-11-01'
code = '000001'
stock = ts.get_hist_data(code, start=begin_time, end=end_time)
stock = stock.sort_index(0) #数据按照日期排序
stock.to_pickle("stock_data_000001.pickle")
print("finish save...")

#读取股票数据
stock = pd.read_pickle("stock_data_000001.pickle")
#5周期,10周期,20周期,60周期
#周线,半月线,月线,季度线
stock["5d"]=stock["close"].rolling(window = 5),mean() #week
stock["10d"]=stock["close"].rolling(window = 10),mean() #half month
stock["20d"]=stock["close"].rolling(window = 20),mean() #month
stock["60d"]=stock["close"].rolling(window = 60),mean() #season

#print(stock.head(1))
#展示股票收盘价信息
stock[["close","5d","10d","20d","60d"]].plot(figsize=(20,10),grid=True)
plt.show()

stock["5-10d"] = stock["5d"] - stock["10d"] #周-半月线差
stock["5-20d"] = stock["5d"] - stock["10d"] #周-月线差
stock[["close","5-10d","5-20d"]].plot(subplots = True, style='b', figsize(20,10), grid = True)
plt.show()

#计算股票收益
#return股票收益= 当日收盘价格 / 钱交易日收盘价格, 然后用Log将数据转为正负数. 负数 = 股票下跌, 正数 = 股票上涨
stock["return"] = np.log(stock["close"] / stock["close"].shift(1))
# stock["return_a"] = stock["close"] / stock["close"].shift(1)
# print(stock[["return","return_a"]].head(15))
stock[["close","return"]].plot(subplots = True, style = 'b', figsize=(20,10), grid=True)
plt.show()

#计算股票的收益率的移动历史标准差
mov_day = int(len(stock)/20)
stock["mov_vol"] = stock["return"].rolling(window = mov_day).std()*math.sqrt(mov_day)
#print(stock["mov_vol"].head(mov_day+1))

stock[["close","mov_vol","return"]].plot(subplots=True, style='b', figsize=(20,10), grid=True)
plt.show()

#print(stock[["mov_vol","return"]].tail(30))
#print(stock["mov_vol"].tail(5).sum())
#print(stock["mov_vol"].tail(10).sum())
#print(stock["mov_vol"].tail(15).sum())
#print(stock["mov_vol"].describe())












