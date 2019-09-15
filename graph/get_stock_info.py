import sqlite3
import numpy as np

#外部函数看main里的

#内部函数
def get_close_price(stock_code,start_date,end_date):
    """
    查询从start_date到end_date的收盘价
    :param stock_code: 股票价格
    :param start_date: 起始日期
    :param end_date: 结束日期
    :return: 列表
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute(
        'select date,close from {} where date>={} and date <={}'.format(stock_code, '"' + start_date + '"',
                                                                        '"' + end_date + '"'))
    dates_prices = []
    for row in result:
        dates_prices.append(row)
    conn.close()
    return dates_prices

#内部函数
def get_N_days_close(stock_code,end_date,day_num=10):
    """
    获得从end_date向前day_num天的收盘价
    :param stock_code: 股票代码
    :param end_date: 结束日
    :param day_num: 天数
    :return: 列表
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute('select date,close from {} where date <= {} order by date desc'.format(stock_code, '"' + end_date + '"'))
    count = 0
    dates_prices=[]
    for row in result:
        dates_prices.append(row)
        count+=1
        if count == day_num:
            break
    conn.close()
    return list(reversed(dates_prices))

def portfolio_history_return(portfolio,shares,start_date,end_date):
    """
    portfolio历史收益率 不包括start_date
    :param portfolio: 股票代码的列表
    :param shares: 每个股票的股数
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 列表
    """
    prices = []
    for stock_code in portfolio:
        prices.append(get_close_price(stock_code,start_date,end_date))
    total=[]
    dates=[]
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j]*prices[j][i][1]
        total.append(sum)

    returns=[]
    for i in range(1,len(total)):
        pre = total[i-1]
        cur = total[i]
        returns.append((cur-pre)/pre)

    return dates[1:],returns


def portfolio_history_vol(portfolio,shares,start_date,end_date):
    """
    portfolio历史波动率 不包括start_date
    :param portfolio: 股票代码的列表
    :param shares: 每个股票的股数
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 列表
    """
    prices=[]
    for stock_code in portfolio:
        prices1=get_N_days_close(stock_code,start_date,10)[:-1]
        prices1.extend(get_close_price(stock_code,start_date,end_date))
        prices.append(prices1)

    total=[]
    dates=[]
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j]*prices[j][i][1]
        total.append(sum)

    returns = []
    for i in range(1, len(total)):
        pre = total[i - 1]
        cur = total[i]
        returns.append((cur - pre) / pre)

    vols=[]
    for i in range(9,len(returns)):
        vol=np.nanstd(np.array(returns[i-9:i+1]))
        vols.append(vol)
    return dates[10:],vols


def pred_portfolio_var(portfolio,shares,date):
    """
    计算portfolio在date的VaR
    :param portfolio: 股票代码的列表
    :param shares: 每个股票的股数
    :param date: 日期
    :return: float实数
    """
    prices=[]
    for stock_code in portfolio:
        prices.append(get_N_days_close(stock_code,date))

    total = []
    dates = []
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j] * prices[j][i][1]
        total.append(sum)

    returns = []
    for i in range(1, len(total)):
        pre = total[i - 1]
        cur = total[i]
        returns.append(np.log(cur/pre))

    sigma = np.std(returns)
    percentile95 = 1.6499
    return95 = np.exp(returns[-1] - percentile95 * sigma)
    return total[-1] - total[-1]*return95

def pred_stock_vol(stock_code,date):
    """
    从数据库查询单只股票的预测波动率
    :param stock_code: 股票代码
    :param date: 日期
    :return: float实数
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute('select EST_V from from ESTSTK where TRADECODE = ? and DATE = ?',(stock_code,date,))
    for row in result:
        pred = float(row[0])
        break
    conn.close()
    return pred


def pred_stock_return(stock_code,date,method = 'lstm'):
    """
    从数据库查询单只股票的预测收益率
    :param stock_code: 股票代码
    :param date: 日期
    :param method: 默认lstm 否则cnn
    :return: float实数
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    if method == 'lstm':
        result = cursor.execute('select EST_R from ESTSTK where TRADECODE = ? and DATE = ?',(stock_code,date,))
        for row in result:
            pred=float(row[0])
            break
        conn.close()
        return pred
    else: #cnn预测法
        result = cursor.execute('select EST_R_2 from ESTSTK where TRADECODE = ? and DATE = ?',(stock_code,date,))
        for row in result:
            pred = float(row[0])
            break
        conn.close()
        return pred


def pred_portfolio_return(portfolio,shares,date,method = 'lstm'):
    """
    预测portfolio收益率
    :param portfolio: 股票代码的列表
    :param shares: 每个股票的股数
    :param date: 日期
    :param method: 默认lstm 否则lstm
    :return: float实数
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    prices=[]
    for stock_code in portfolio:
        result = cursor.execute('select close from {} where date <= {}'.format(stock_code,'"'+date+'"'))
        for row in result:
            prices.append(row[0])
            break
    conn.close()
    pred_returns=[]
    for stock_code in portfolio:
        pred_returns.append(pred_stock_return(stock_code,date,method))
    cur_total = 0
    pred_total = 0
    for i in range(len(shares)):
        cur_total += shares[i]*prices[i]
        pred_total += shares[i]*prices[i]*(1+pred_returns[i])

    return (pred_total-cur_total)/cur_total

if __name__=='__main__':
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
    portfolio=['SZ000002','SZ000004']
    shares=[100,200]
    print(pred_stock_return('SZ000002','2018-01-02'))
    print(pred_portfolio_return(portfolio,shares,'2018-01-02','lstm'))








