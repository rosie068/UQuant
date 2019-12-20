import numpy as np
import pandas as pd
import datetime as dt
from pandas import Series, DataFrame, isnull
from datetime import timedelta, datetime
from CAL.PyCAL import *

##因子回测
lcap = DataAPI.MktStockFactorsOneDayGet(secID=set_universe('HS300'),tradeDate=u"20160922",field=['secID','LCAP'],pandas="1").set_index('secID')
#lcap.head()

# 去极值winsorize
after_winsorize = winsorize(lcap['LCAP'].to_dict())
lcap['winsorized LCAP'] = np.nan
lcap.loc[after_winsorize.keys(),'winsorized LCAP'] = after_winsorize.values() 
#print(lcap)

# 标准化standardize
after_standardize = standardize(lcap['winsorized LCAP'].to_dict())
lcap['standardized LCAP'] = np.nan
lcap.loc[after_standardize.keys(),'standardized LCAP'] = after_standardize.values()
#lcap['standardized LCAP'].plot(figsize=(14,5))
#print(lcap)

start = '2013-01-01'
end = '2016-09-01'
benchmark = 'HS300'
universe = DynamicUniverse('HS300') + DynamicUniverse('ZZ500')
capital_base = 10000000
freq = 'd'  # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = Monthly(1)

accounts = {'fantasy_account': AccountConfig(account_type='security',capital_base=10000000)}

def initialize(context):
    context.signal_generator = SignalGenerator(Signal('LCAP'))

def handle_data(context):
    universe = context.get_universe()
    yesterday = context.previous_date.strftime('%Y-%m-%d')
    data = context.history(universe, ['LCAP'], time_rage = 1, style = 'tas')
    data = data[yesterday]

    factor = data['LCAP'].dropna()
    factor = pd.Series(winsorize(factor, win_type='QuantileDraw',pvalue=0.05))
    factor = 1.0/factor
    factor = factor.replace([np.inf, -np.inf], 0.0)
    signal_lcap = standardize(dict(factor))
    
    signal = pd.Series(signal_lcap)
    
    wts = simple_long_only(signal, yesterday)
    account = context.get_account('fantasy_account')
    current_position = account.get_positions(exclude_halt=True)
    target_position = wts.keys()
    
    for stock in set(current_position).difference(target_position):
        account.order_to(stock, 0)
    for stock in target_position:
        account.order_pct_to(stock, wts[stock])
