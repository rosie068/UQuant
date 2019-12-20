import numpy as np
import pandas as pd
import datetime as dt
from pandas import Series, DataFrame, isnull
from datetime import timedelta, datetime
from CAL.PyCAL import *
from statsmodels import regression
import statsmodels.api as sm

today = Date.todaysDate()
begin_date = (today - Period('1Y')).toDateTime().strftime('%Y%m%d')
end_date = today.toDateTime().strftime('%Y%m%d')

#DataAPI.IdxGet(secID=u"",ticker=u"399300",field=u"",pandas="1")
market = DataAPI.MktIdxdGet(indexID=u"",ticker=u"399001",tradeDate=u"",beginDate=begin_date,endDate=end_date,exchangeCD=u"XSHE",field=['ticker','tradeDate','preCloseIndex','closeIndex','CHG','CHGPct'],pandas="1")
market = market.dropna()
#print(market)

stock = DataAPI.MktEqudGet(secID=u"",ticker=u"000001",tradeDate=u"",beginDate=begin_date,endDate=end_date,isOpen=u"",field=['ticker','tradeDate','actPreClosePrice','closePrice'],pandas="1")
stock = stock.dropna()
#print(stock)

risk_free = DataAPI.MktRefIrGet(secID=u"",ticker=u"FR001",beginDate=begin_date,endDate=end_date,field=['ticker','tradeDate','rate'],pandas="1")
risk_free = risk_free.dropna()
#print(risk_free)

###CAPM模型有，Rs=Rf+βs∗(Rm−Rf)。式中，Rs表示股票收益，Rf表示无风险收益率，Rm表示市场收益
market['return']=(market['closeIndex']-market['preCloseIndex'])
market['return'] = (market['return']/market['closeIndex'])
market = market.drop([0])
#print(market)

stock['return']=(stock['closePrice']-stock['actPreClosePrice'])
stock['return'] = (stock['return']/stock['closePrice'])
stock = stock.drop([0])
#print(stock)

risk_free['preRate'] = risk_free['rate']
for index in range(risk_free.shape[0]):
    if index == 0:
        risk_free.iat[index,3] = 0
    else:
        risk_free.iat[index,3] = risk_free.iat[index-1,2]
risk_free['return']=(risk_free['rate']-risk_free['preRate'])
risk_free['return'] = (risk_free['return']/risk_free['rate'])
risk_free = risk_free.drop([0])
#print(risk_free)

market = market.dropna()
stock = stock.dropna()
risk_free = risk_free.dropna()

#############linear regression Rs=Rf+βs∗(Rm−Rf) -> Rs-Rf=βs∗(Rm−Rf)
y = np.array(stock['return']-risk_free['return'])
y = y[1:len(y)-10]
Y = y.T
#print(len(Y))

X = np.array(market['return']-risk_free['return'])
#X = np.column_stack(Rm_Rf)
##################I had to adjust the length of X so regression worked, but NOT ok in the long run
X = X[1:len(y)+1]
#print(len(X))

X = sm.add_constant(X)
mod = regression.linear_model.OLS(Y, X).fit()
a = mod.params
#print 'beta, residual = ', a

########αs=Rs−[Rf+βs∗(Rm−Rf)]
beta = a[0]
stock['alpha'] = stock['return']-risk_free['return']-beta*(market['return']-risk_free['return'])
#print(stock['alpha'].mean())

print '沪深000001平安保险收益率 = (',stock['alpha'].mean(), '+ Rf) +', beta, '* (Rm-Rf)'
