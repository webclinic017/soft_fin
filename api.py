from pandas import Series

'''期权期货函数接口'''
import backtest.options

def portfolio_delta(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str) -> Series:
    return backtest.options.portfolio_delta(asset_id, asset_mount, cash, begin_t, end_t)


def portfolio_gamma(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str) -> Series:
    return backtest.options.portfolio_gamma(asset_id, asset_mount, cash, begin_t, end_t)


def portfolio_vega(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str) -> Series:
    return backtest.options.portfolio_vega(asset_id, asset_mount, cash, begin_t, end_t)


def portfolio_rho(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str) -> Series:
    return backtest.options.portfolio_rho(asset_id, asset_mount, cash, begin_t, end_t)


def portfolio_theta(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str) -> Series:
    return backtest.options.portfolio_theta(asset_id, asset_mount, cash, begin_t, end_t)


def portfolio_volatility(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str, time: int=10) -> Series:
    return backtest.options.portfolio_volatility(asset_id, asset_mount, cash, begin_t, end_t, time)


def portfolio_earning_rate(asset_id: list, asset_mount: list, cash: float, begin_t: str, end_t: str,
                           time: int=10) -> Series:
    return backtest.options.portfolio_earning_rate(asset_id, asset_mount, cash, begin_t, end_t, time)


def retrain_delta_model(protfolio_id: str, asset_id: list, asset_mount: list, cash: float, options: str, test: int = 0)->Series:
    return backtest.options.retrain_delta_model(protfolio_id, asset_id, asset_mount, cash, options, test)


def retrain_gamma_model(protfolio_id: str, asset_id: list, asset_mount: list, cash: float, options1:str, options2:str, test:int=0)->Series:
    return backtest.options.retrain_gamma_model(protfolio_id, asset_id, asset_mount, cash, options1, options2, test)


def fit_delta(protfolio_id: str, asset_id: list, asset_mount: list, cash: float, options, begin_t: str, end_t: str,test:int=0) -> Series:
    return backtest.options.fit_delta(protfolio_id, asset_id, asset_mount, cash, options, begin_t, end_t, test)


def fit_gamma(protfolio_id: str, asset_id: list, asset_mount: list, cash: float, options1:str, options2:str, begin_t:str, end_t:str, test:int=0)->Series:
    return backtest.options.fit_gamma(protfolio_id, asset_id, asset_mount, cash, options1, options2, begin_t, end_t,
                                      test)


def cal_option_amt(total_value: float, option:str, portion: float, t1:str) -> int:
    return backtest.options.cal_option_amt(total_value, option, portion, t1)


def generate_recommend_option_delta(protfolio_id:str, asset_id:list, asset_mount:list, cash:float)->list:
    return backtest.options.generate_recommend_option_delta(protfolio_id, asset_id, asset_mount, cash)


def generate_recommend_option_gamma(protfolio_id:str, asset_id:list, asset_mount:list, cash:float)->list:
    return backtest.options.generate_recommend_option_gamma(protfolio_id, asset_id, asset_mount, cash)


def get_portfolio_beta(asset_id, weight_list):
    return backtest.options.get_portfolio_beta(asset_id, weight_list)


def retrain_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num=0):
    return backtest.options.retrain_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num)


def train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num=0):
    return backtest.options.train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num)


def fit_beta(protfolio_id, asset_id, asset_mount, cash, futures, test=0):
    return backtest.options.fit_beta(protfolio_id, asset_id, asset_mount, cash, futures, test=0)


def cal_option_change_rate(option, t):
    return backtest.options.cal_option_change_rate(option, t)


def cal_future_change_rate(future, t):
    return backtest.options.cal_future_change_rate(future, t)


def generate_recommend_option_delta(protfolio_id, asset_id, asset_mount, cash):
    return backtest.options.generate_recommend_option_delta(protfolio_id, asset_id, asset_mount, cash)

def cal_future_amt(total_value,futures,portion,t1):
    return backtest.options.cal_future_amt(total_value,futures,portion,t1)

def generate_recommend_future(protfolio_id,asset_id,asset_mount,cash):
    return backtest.options.generate_recommend_future(protfolio_id,asset_id,asset_mount,cash)


#print(get_portfolio_beta(['000001.SZ', '000010.SZ'], [100, 100]))

'''定期调整和条件触发'''
###以下是定期调整和条件触发
import adjust.triggers


def stock_least_position(user_position: int, setting: int) -> bool:
    return adjust.triggers.stock_least_position(user_position, setting)


def stock_volatility(stock_code: str, time: int, setting: float) -> bool:
    return adjust.triggers.stock_volatility(stock_code, time, setting)


def stock_mean_return(stock_code: str, time: int, setting: float) -> bool:
    return adjust.triggers.stock_mean_return(stock_code, time, setting)


def stock_change(stock_code: str, time: int, top: float, bottom: float) -> bool:
    return adjust.triggers.stock_change(stock_code, time, top, bottom)


###### 以上是定期调整与条件触发的共有函数 ######

def stock_turnover_rate(stock_code: str, top: float, bottom: float) -> bool:
    return adjust.triggers.stock_turnover_rate(stock_code, top, bottom)


def stock_macd(stock_code: str, top: float, bottom: float) -> bool:
    return adjust.triggers.stock_macd(stock_code, top, bottom)


def stock_rsi(stock_code: str, rsi_time: int, top: float, bottom: float) -> bool:
    return adjust.triggers.stock_rsi(stock_code, rsi_time, top, bottom)


def stock_kdj(stock_code: str, K: float, D: float, J: float) -> bool:
    return adjust.triggers.stock_kdj(stock_code, K, D, J)


def stock_sharpe(stock_code: str, setting: float) -> bool:
    return adjust.triggers.stock_sharpe(stock_code, setting)
    # 获取10日收益率、波动率


###### 以上是条件触发的指标函数 ######
def portfolio_var(portfolio: str, setting: float) -> bool:
    return adjust.triggers.portfolio_var(portfolio, setting)


def portfolio_volatility(portfolio: str, cash: float, setting: float) -> bool:
    return adjust.triggers.portfolio_volatility(portfolio, cash, setting)


def portfolio_diff(portfolio_id: str, portfolio: str, cash: float, alpha: float, top: float) -> bool:
    return adjust.triggers.portfolio_diff(portfolio_id, portfolio, cash, alpha, top)

'''风格因子'''
import multi.api
def get_size(ts_code: str, fields: str):
    return multi.api.get_size(ts_code, fields)

def get_momentum(ts_code: str):
    return multi.api.get_size(ts_code, fields)

def get_book_to_market(stock_code: str):
    return multi.api.get_book_to_market(stock_code)

def get_liquidity(stock_code: str):
    return multi.api.get_liquidity(stock_code)
    

'''绘图'''
    #外部接口
    #预测股票收益率 此处加了功能 如果不是lstm 就是cnn
    # float_a=pred_stock_return('SH600717', '2018-01-02','lstm')
    #
    # #预测股票波动率
    # float_b=pred_stock_vol('SH600717', '2018-01-02')
    # portfolio = ['SH600000','SH600717']
    # shares=[100,200]
    #
    # #预测组合收益率
    # float_c=pred_portfolio_return(portfolio,shares,'2018-01-02')
    #
    # #计算组合VaR
    # float_d=pred_portfolio_var(portfolio,shares,'2018-01-02')
    #
    # #计算组合历史收益率 不包括起始日
    # dates_list_a,floats_list_a = portfolio_history_return(portfolio,shares,'2018-01-02','2019-06-12')
    #
    # # 计算组合历史波动率 不包括起始日
    # dates_list_b,floats_list_b = portfolio_history_vol(portfolio,shares,'2018-01-02','2019-06-12')
import graph.get_stock_info
def portfolio_history_return(portfolio,shares,start_date,end_date):
    return graph.get_stock_info.portfolio_history_return(portfolio,shares,start_date,end_date)

def portfolio_history_vol(portfolio,shares,start_date,end_date):
    return graph.get_stock_info.portfolio_history_vol(portfolio,shares,start_date,end_date)

def pred_portfolio_var(portfolio,shares,date):
    return graph.get_stock_info.pred_portfolio_var(portfolio,shares,date)

def pred_stock_vol(stock_code,date):
    return graph.get_stock_info.pred_stock_vol(stock_code,date)

def pred_stock_return(stock_code,date,method):
    return graph.get_stock_info.pred_stock_return(stock_code,date,method)

def pred_portfolio_return(portfolio,shares,date,method):
    return graph.get_stock_info.pred_portfolio_return(portfolio,shares,date,method)

import sqlite3
def get_all_stocks():
    conn = sqlite3.connect('data/fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute('select name from sqlite_master where type="table" order by name')
    all = []
    for row in result:  
        if len(row[0])==8:
            if row[0][:2]=='SH' or row[0][:2]=='SZ':
                all.append(row[0])
    conn.close()
    return all
    
def get_stock_histroy(stock_id):
    conn = sqlite3.connect('data/fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute('select date,open,high,low,close,volume,amt from '+stock_id)
    tmp=[]
    cursor = conn.cursor()
    name=cursor.execute('select name from stockinfo where tradecode="'+stock_id[-6:]+"."+stock_id[:2]+'"')
    for i in name:
        name_new=i
    for i in result:
        line=(i[0],str(i[1]),str(i[2]),str(i[3]),str(i[4]),str(i[5]),str(i[6]))
        tmp.append(line)
        #print(str(i+name_new))
    print(tmp)
    return tmp


'''手续费计算'''
import API.Poundage
def stock_poundage(DealAmount, BuyIn=True, BCRate=0.002):
    return API.Poundage.stock_poundage(DealAmount, BuyIn, BCRate)

def future_poundage(DealAmount): 
    return API.Poundage.future_poundage(DealAmount)

def options_poundage(NumOfPiece, BCUnitPrice=5): 
    return API.Poundage.options_poundage(NumOfPiece, BCUnitPrice)

'''推荐投资组合'''
import e.recommend_portfolio
def recommend_portfolio(beta_threshold, stock_num=10, return_level='low'):
    return e.recommend_portfolio.recommend_portfolio(beta_threshold, stock_num, return_level)
if __name__ == "__main__":
    get_stock_histroy("SZ000002")

# %%================ 此api调用示例 =======================

#rec_portfolio = recommend_portfolio(beta_threshold=np.full(factor_num, 0.45), stock_num=30, return_level='mid')
# 说明： 均为可选参数，找不到合适解或个股数目要求太多时会提示错误类型
# 提醒：当return_level设为'high'时，阈值最好不低于0.45，否则可能找不到合适解
# beta_threshold 默认np.full(factor_num,0.4)
# stock_num 默认10，不超过32
# return_level 默认'low'
#print(rec_portfolio)

