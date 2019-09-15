from lstm_predict import lstm_pred
from get_stock_info import portfolio_history_vol
from CNN import final_cnn_predict
from LSTM import final_lstm_predict
import numpy as np
import sqlite3

def calc_pred_stock_return(stock_code, date):
        """
        stock_code是形如 SH600000 的字符串
        date是形如 2019-06-01 的字符串
        预测date日期后一天stock_code股票的收益率
        将预测结果存于数据库中
        :param stock_code: 股票代码
        :param date: 日期
        :return: None
        """
        conn = sqlite3.connect('fin_set.db')
        cursor = conn.cursor()
        string1=[]
        string2=[]
        string3=[]
        for i in range(1,8):
            string1.append('F'+str(i))
        for i in range(1,520):
            string2.append('B'+str(i))

        for i in range(8,520):
            string3.append('F'+str(i))

        #获得三个查询列表的的名字
        cmd1 = ','.join(string1)
        cmd2 = ','.join(string2)
        cmd3 = ','.join(string3)
        result = cursor.execute(
            'select  {}  from {} where date <= {}'.format(cmd1,stock_code,'"' + date + '"'))
        count = 0
        factor1=[[] for i in range(5)]
        factor2=[[] for i in range(514)]
        beta=[0 for i in range(519)]
        for row in result:
            for i in range(5):
                factor1[i].append(row[i])
            for i in range(2):
                factor2[i].append(row[5+i])
            count += 1
            if count == 40:
                break

        result = cursor.execute(
            'select  {}  from BETA '.format(cmd2))

        for row in result:
            for i in range(520):
                beta[i] = row[i]

        result = cursor.execute(
            'select  {}  from MARKET '.format(cmd3))
        for row in result:
            for i in range(512):
                factor2[i+2].append(row[i])

        result = cursor.execute('select RF from MARKET where date = {}'.format('"' + date + '"'))
        rf = 0
        for row in result:
            rf = row[0]
            break

        #万一预测模型未收敛 则需要处理异常
        try:
            #用lstm预测
            pf1=final_lstm_predict(factor1)
            #用cnn预测
            pf2=final_cnn_predict(factor1)

            est_r1=rf
            est_r2=rf

            for i in range(5):
                est_r1+=pf1[i]*beta[i]
                est_r2+=pf2[i]*beta[i]

            for i in range(514):
                est_r1 += factor2[i][-1] * beta[i+5]
                est_r2 += factor2[i][-1] * beta[i+5]
        except:
            est_r1 = np.random.normal(loc=rf, scale=0.03, size=None)
            est_r2 = np.random.normal(loc=rf, scale=0.03, size=None)

        #将两种预测的结果存在数据库中
        cursor.execute('update ESTSTK set EST_R = {} where TRADECODE = {} and date = {}'.format(est_r1,stock_code,'"' + date + '"',))
        cursor.execute('update ESTSTK set EST_R_2 = {} where TRADECODE = {} and date = {}'.format(est_r2, stock_code,
                                                                                                '"' + date + '"', ))
        conn.commit()
        conn.close()


def calc_pred_stock_vol(stock_code,date):
    """
    将该股票的波动率预测存于数据库中
    :param stock_code: 股票名称
    :param date: 日期
    :return: None
    """
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute(
        'select date,close from {} where date <={} order by date desc'.format(stock_code, '"' + date + '"'))
    count = 0
    dates_prices = []
    for row in result:
        dates_prices.append(row)
        count += 1
        if count == 31:
            break

    start_date = dates_prices[-1][0]
    portfolio=[stock_code]
    shares=[1]
    #计算历史波动率
    d,history_vols = portfolio_history_vol(portfolio,shares,start_date,date)

    #用一维lstm预测历史波动率
    ls = lstm_pred(history_vols)
    pred_vol = ls.predict()
    cursor.execute('update ESTSTK set EST_R = {} where TRADECODE = {} and date = {}'.format(pred_vol,stock_code,'"' + date + '"',))
    conn.commit()
    conn.close()

def update_database_daily(stock_pool,date):
    """
    每日更新预测数据库
    :param stock_pool: 所有股票池
    :param date: 日期
    """
    #对每个股票更新预测表
    for stock_code in stock_pool:
        calc_pred_stock_vol(stock_code,date)
        calc_pred_stock_return(stock_code,date)



if __name__ == '__main__':
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    # result = cursor.execute('select name from sqlite_master where type="table" order by name')
    # all = []
    # for row in result:
    #     all.append(row[0])
    #
    # for stock_code in all:
    #     result = cursor.execute('select * from {} where date = {} or date = {}'.format(stock_code,'"'+start+'"','"'+end+'"'))
    #     count = 0
    #     for row in result:
    #         count+=1
    #     if count == 2:
    #         stock_pool.append(stock_code+'\n')
    #
    # print(stoc k_pool)
    # conn.close()
    # f=open('stock_pool.txt','w')
    # f.writelines(stock_pool)
    # f.close()
    res = cursor.execute('select * from SH600717 where date = ? ',(start,))
    for row in res:
        print(row)
