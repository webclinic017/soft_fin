import tushare as ts
import pandas as pd
import numpy as np
import talib
import sqlite3
import json
import options
import graph.get_stock_info as gsi# 自定义py文件
def get_stock_info(stock_code: str, time: int, column:list) -> pd.DataFrame:
    cnx = sqlite3.connect('fin_set.db')
    df = pd.read_sql_query("SELECT "+','.join(column)+" FROM "+stock_code+" ORDER BY DATE DESC LIMIT "+str(time), cnx)
    return df[::-1]

def cal_mrt(price:list) -> float:
    return np.mean(np.diff(np.log(price)))

def get_mrt(stock_code:str, time: int) -> float:
    pr = list(get_stock_info(stock_code, time + 1,['CLOSE'])['CLOSE'])
    return cal_mrt(pr)

def cal_vol(price:list) -> float:
    rt = np.diff(np.log(price))
    vol = np.std(rt)
    return vol

def get_vol(stock_code:str, time: int) -> float:
    pr = list(get_stock_info(stock_code, time + 1,['CLOSE'])['CLOSE'])
    return cal_vol(pr)

###### 内部使用的函数 ######
def stock_least_position(user_position: int, setting: int) -> bool:
    return user_position < setting

def stock_volatility(stock_code: str, time: int, setting: float) -> bool:
    vol = get_vol(stock_code, time)
    print(vol)
    return vol > setting

def stock_mean_return(stock_code: str, time: int, setting: float) -> bool:
    mrt = get_mrt(stock_code, time)
    print(mrt)
    return mrt < setting

def stock_change(stock_code: str, time: int, top: float, bottom: float) -> bool:
    pr = get_stock_info(stock_code, time, ['CLOSE'])['CLOSE']
    cls = pr[time - 1]
    pre_cls = pr[0]
    change = (cls - pre_cls) / pre_cls
    print(change)
    return change > top or change < bottom

###### 以上是定期调整与条件触发的共有函数 ######

def stock_turnover_rate(stock_code: str, top: float, bottom: float) -> bool:
    tr = list(get_stock_info(stock_code, 1, ['TURN'])['TURN'])[0] # 需要当天的换手率
    print(tr)
    return tr > top or tr < bottom

def stock_macd(stock_code: str,  top: float, bottom: float) -> bool:
    pr = get_stock_info(stock_code, 27, ['CLOSE'])['CLOSE']
    close = [float(x) for x in pr]
    macd = talib.MACD(np.array(close), fastperiod=6, slowperiod=12, signalperiod=9)[0][-1] # 最近一天的MACD
    print(macd)
    return macd > top or macd < bottom

def stock_rsi(stock_code: str, rsi_time:int, top:float, bottom:float) -> bool:
    if rsi_time not in [6,12,24]:
        print("WARNING: rsi must be calculated for time 6, 12, 24, but get", rsi_time)
        
    pr = get_stock_info(stock_code, 25,['CLOSE'])['CLOSE']
    close = [float(x) for x in pr]
    rsi = talib.RSI(np.array(close), timeperiod=rsi_time)[-1]
    print(rsi)
    return rsi > top or rsi < bottom
    
def stock_kdj(stock_code: str, K: float, D: float, J:float) -> bool:
    data = get_stock_info(stock_code, 9, ['CLOSE','HIGH','LOW'])
    ret = {}
    low_list = data['LOW'].rolling(9, min_periods=1).min()
    high_list = data['HIGH'].rolling(9, min_periods=1).max()
    rsv = (data['CLOSE'] - low_list) / (high_list - low_list) * 100
    data['K'] = rsv.ewm(com=2, adjust=False).mean()
    data['D'] = data['K'].ewm(com=2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']
    ret['K'] = data['K'][0]
    ret['D'] = data['D'][0]
    ret['J'] = data['J'][0]
    print(ret)
    return ret['K'] > K or ret['D'] > D or ret['J'] > J

def stock_sharpe(stock_code: str, setting: float) -> bool:
    # 获取10日收益率、波动率
    mrt = get_mrt(stock_code, 10)
    vol = get_vol(stock_code, 10)
    print(mrt/vol)
    return (mrt / vol) < setting


###### 以上是条件触发的指标函数 ######

def json2list(jsonstr: str, ts_format = True):
    pf = json.loads(jsonstr)
    stock = []
    share = []
    option = ''
    optshare = 0
    for i in pf:
        if i[:2] in ['IC','IF','IH']:
            if ts_format:
                option = i[2:]+'.'+i[:2]
            else:
                option = i
            optshare = pf[i]
        else:
            if ts_format:
                stock.append(i[2:]+'.'+i[:2])
            else:
                stock.append(i)
            share.append(pf[i])
    return stock, share, option, optshare

def portfolio_var(portfolio: str, setting: float) -> bool:
    pf, sh, option, optshare = json2list(portfolio,ts_format=False)
    date = get_stock_info('SZ000001', 1, ['DATE'])['DATE'][0]
    var = gsi.pred_portfolio_var(pf,sh,date)
    print(var)
    return var > setting
    
def portfolio_mean_return(portfolio: str, time:int, top: float, bottom: float) -> bool:
    pf, sh, option, optshare = json2list(portfolio)
    date = list(get_stock_info('SZ000001', time+1, ['DATE'])['DATE'])
    vals = options.portfolio_total_value(pf,sh,0,date[0],date[-1])
    return not bottom < cal_mrt(vals) < top
    
def portfolio_change(portfolio: str, time:int, top: float, bottom: float) -> bool:
    pf, sh, option, optshare = json2list(portfolio)
    date = list(get_stock_info('SZ000001', time+1, ['DATE'])['DATE'])
    vals = options.portfolio_total_value(pf,sh,0,date[0],date[-1])
    return not bottom < (vals[-1]-vals[0])/vals[0] < top

def portfolio_volatility(portfolio: str,time:int, cash: float, setting: float) -> bool:
    pf, sh, option, optshare = json2list(portfolio)
    date = get_stock_info('SZ000001', 1, ['DATE'])['DATE'][0]
    vol = options.portfolio_volatility(pf,sh,cash, begin_t = date, end_t = date,time=10)
    return vol > setting

###### 以上是组合类指标，超限时股票全部卖出 ######

def portfolio_diff(portfolio_id:str, portfolio: str, cash:float, alpha:float, top: float) -> bool:
    pf, sh, option, optshare = json2list(portfolio)
    date = get_stock_info('SZ000001', 1, ['DATE'])['DATE'][0]
    a=options.fit_delta(portfolio_id, pf, sh, cash, option, date, date)
    need=options.cal_option_amt(options.portfolio_total_value(pf,sh,cash,date,date)[-1], option, alpha * a[0])
    return abs(optshare - need > top)

###### 以上是期权期货类触发指标 ######

    