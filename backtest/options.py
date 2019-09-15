# -*- coding: utf-8 -*-
# author Yuan Manjie
# date 2019/9/5

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('data/fin_set.db')  # 连接到数据库
c = conn.cursor()  # 创建游标

pd.set_option('mode.use_inf_as_na', True)

# 常量定义
contract_unit = dict()
contract_unit['50ETF'] = 10000
contract_unit['IF'] = 300
contract_unit['IC'] = 200
contract_unit['IH'] = 300

# 输入id找不到时引起的自定义错误


class ID_Not_Exist_ERROR(Exception):
    def __init__(self, id):
        self.id = id


def format_date(date):
    pd_t = pd.Timestamp(date)
    return str(pd_t)[:10]


# 获取根据id获取一段时期内证券的数据
def get_stock_data(id, begin_t='', end_t='', t_before=0):
    new_id = id[-2:] + id[:6]
    try:
        if begin_t != ''and end_t != '':
            b_t = format_date(begin_t)
            e_t = format_date(end_t)
            sql = "select \"index\" from " + new_id + \
                " where DATE>='" + b_t + "' and DATE<='" + e_t + "'"
            sql_dat = list(c.execute(sql))
            b_i = sql_dat[0][0]
            e_i = sql_dat[-1][0]
            sql = "select DATE,OPEN from " + new_id + " where \"index\">='" + \
                str(b_i - t_before) + "' and \"index\"<='" + str(e_i) + "'"
            sql_dat = pd.DataFrame(list(c.execute(sql)),
                                   columns=['date', 'open'])
            sql_dat.index = map(pd.Timestamp, sql_dat['date'])
        else:
            sql = "select DATE,OPEN from " + new_id
            sql_dat = pd.DataFrame(list(c.execute(sql)),
                                   columns=['date', 'open'])
            sql_dat.index = map(pd.Timestamp, sql_dat['date'])
        return sql_dat.iloc[:, 1:]
    except:
        raise ID_Not_Exist_ERROR(id)

# 获取期货数据


def get_futures_data(id, begin_t='', end_t='', t_before=0):
    new_id = id
    try:
        if type(begin_t) == type(1):
            sql = "select \"index\" from " + new_id
            sql_dat = list(c.execute(sql))
            b_i = sql_dat[begin_t][0]
            e_i = sql_dat[end_t][0]
            sql = "select * from " + new_id + " where \"index\">='" + \
                str(b_i - t_before) + "' and \"index\"<='" + str(e_i) + "'"
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=['index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE',
                                                                  'SETTLE', 'VOLUME', 'AMT', 'OI', 'VWAP', 'OI_CHG', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO', 'VOLATILITYRATIO', 'US_IMPLIEDVOL'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        if begin_t != ''and end_t != '':
            b_t = format_date(begin_t)
            e_t = format_date(end_t)
            sql = "select \"index\" from " + new_id + \
                " where DATE>='" + b_t + "' and DATE<='" + e_t + "'"
            sql_dat = list(c.execute(sql))
            b_i = sql_dat[0][0]
            e_i = sql_dat[-1][0]
            sql = "select * from " + new_id + " where \"index\">='" + \
                str(b_i - t_before) + "' and \"index\"<='" + str(e_i) + "'"
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=[
                                   'index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE', 'VOLUME', 'AMT', 'OI', 'OI_CHG', 'VWAP', 'VOLRATIO'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        else:
            sql = "select * from " + new_id
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=[
                                   'index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE', 'VOLUME', 'AMT', 'OI', 'OI_CHG', 'VWAP', 'VOLRATIO'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        return sql_dat
    except:
        raise ID_Not_Exist_ERROR(id)


# 获取期权数据
def get_options_data(id, begin_t='', end_t='', t_before=0):
    new_id = id[-2:] + id[:8]
    try:
        if type(begin_t) == type(1):
            sql = "select \"index\" from " + new_id
            sql_dat = list(c.execute(sql))
            b_i = sql_dat[begin_t][0]
            e_i = sql_dat[end_t][0]
            sql = "select * from " + new_id + " where \"index\">='" + \
                str(b_i - t_before) + "' and \"index\"<='" + str(e_i) + "'"
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=['index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE',
                                                                  'SETTLE', 'VOLUME', 'AMT', 'OI', 'VWAP', 'OI_CHG', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO', 'VOLATILITYRATIO', 'US_IMPLIEDVOL'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        elif begin_t != ''and end_t != '':
            b_t = format_date(begin_t)
            e_t = format_date(end_t)
            sql = "select \"index\" from " + new_id + \
                " where DATE>='" + b_t + "' and DATE<='" + e_t + "'"
            sql_dat = list(c.execute(sql))
            b_i = sql_dat[0][0]
            e_i = sql_dat[-1][0]
            sql = "select * from " + new_id + " where \"index\">='" + \
                str(b_i - t_before) + "' and \"index\"<='" + str(e_i) + "'"
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=['index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE',
                                                                  'SETTLE', 'VOLUME', 'AMT', 'OI', 'VWAP', 'OI_CHG', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO', 'VOLATILITYRATIO', 'US_IMPLIEDVOL'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        else:
            sql = "select * from " + new_id
            sql_dat = pd.DataFrame(list(c.execute(sql)), columns=['index', 'DATE', 'PRESETTLE', 'PRECLO', 'OPEN', 'HIGH', 'LOW', 'CLOSE',
                                                                  'SETTLE', 'VOLUME', 'AMT', 'OI', 'VWAP', 'OI_CHG', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO', 'VOLATILITYRATIO', 'US_IMPLIEDVOL'])
            sql_dat.index = map(pd.Timestamp, sql_dat['DATE'])
        sql = 'select EXE_PRICE,ENDDATE from OPTIONINFO where TRADECODE=\'' + id + '\''
        sql_tmp = list(c.execute(sql))
        sql_dat['EXE_PRICE'] = sql_tmp[0][0]
        sql_dat['EXE_ENDDATE'] = pd.Timestamp(sql_tmp[0][1])
        return sql_dat
    except Exception as e:
        raise ID_Not_Exist_ERROR(id)


# id类型判读函数
def is_stock(id):
    if id[-3:] in ['.SZ', '.SH'] and len(id) == 9:
        return True
    else:
        return False


def is_futures(id):
    if id[:2] in ['IF', 'IC', 'IH'] and len(id) == 6:
        return True
    else:
        return False


def is_options(id):
    if id[-3:] in ['.SH'] and len(id) == 11:
        return True
    else:
        return False


# 计算历史总净值数据
def portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t, t_before=0):
    total = []
    stocks = []
    keys = []
    stock_keys = []
    for ii, i in enumerate(asset_id):
        if is_stock(i):
            temp = get_stock_data(i, begin_t, end_t, t_before)
            if len(temp) != 0:
                keys += [i]
                stock_keys += [i]
                total += [temp * asset_mount[ii]]
                stocks += [temp * asset_mount[ii]]
        elif is_futures(i):
            temp = get_futures_data(i, begin_t, end_t, t_before)
            if len(temp) != 0:
                keys += [i]
                temp['delta'] = temp['OPEN'].diff(1)
                temp = temp.fillna(0)
                temp['delta'] *= contract_unit[i[:2]]
                temp['delta'][0] += temp['OPEN'][0] * \
                    contract_unit[i[:2]] * 0.08
                for j in range(1, len(temp)):
                    temp['delta'][j] += temp['delta'][j - 1]
                    if temp['delta'][j] <= 0:
                        temp['delta'][j:] = 0
                        break
                total += [temp['delta'] * asset_mount[ii]]
        elif is_options(i):
            temp = get_options_data(i, begin_t, end_t, t_before)
            temp = temp.fillna(0)
            if len(temp) != 0:
                keys += [i]
                total += [temp['OPEN'] * asset_mount[ii] *
                          contract_unit['50ETF']]
    if len(total) == 0:
        return pd.DataFrame()
    else:
        total = pd.concat(total, axis=1, keys=keys)
        total = total.fillna(method='ffill')
        total = total.fillna(method='bfill')
        if len(stocks) == 0:
            stocks = pd.DataFrame()
        else:
            stocks = pd.concat(stocks, axis=1, keys=stock_keys)
            stocks = stocks.fillna(method='ffill')
            stocks = stocks.fillna(method='bfill')
    return total.sum(axis=1) + cash, stocks.sum(axis=1)


# 计算组合各指标值
def portfolio_delta(asset_id, asset_mount, cash, begin_t, end_t):
    total, stock = portfolio_total_value(
        asset_id, asset_mount, cash, begin_t, end_t, 1)
    delta = total.diff(1) / stock.diff(1)
    return delta[1:]


def portfolio_gamma(asset_id, asset_mount, cash, begin_t, end_t):
    total, stock = portfolio_total_value(
        asset_id, asset_mount, cash, begin_t, end_t, 2)
    delta = total.diff(1) / stock.diff(1)
    gamma = delta.diff(1) / stock.diff(1)
    return gamma[2:]


def portfolio_vega(asset_id, asset_mount, cash, begin_t, end_t):
    total_vega = []
    keys = []
    for ii, i in enumerate(asset_id):
        if is_options(i):
            temp = get_options_data(i, begin_t, end_t)
            if len(temp) != 0:
                keys += [i]
                total_vega += [temp['VEGA'] *
                               asset_mount[ii] * contract_unit['50ETF']]
    if len(total_vega) == 0:
        return pd.DataFrame()
    else:
        total = pd.concat(total_vega, axis=1, keys=keys)
        total = total.fillna(0)
        return total.sum(axis=1)


def portfolio_rho(asset_id, asset_mount, cash, begin_t, end_t):
    total_rho = []
    keys = []
    for ii, i in enumerate(asset_id):
        if is_options(i):
            temp = get_options_data(i, begin_t, end_t)
            if len(temp) != 0:
                keys += [i]
                total_rho += [temp['RHO'] * asset_mount[ii] *
                              contract_unit['50ETF']]
    if len(total_rho) == 0:
        return pd.DataFrame()
    else:
        total = pd.concat(total_rho, axis=1, keys=keys)
        total = total.fillna(0)
        return total.sum(axis=1)


def portfolio_theta(asset_id, asset_mount, cash, begin_t, end_t):
    total_theta = []
    keys = []
    for ii, i in enumerate(asset_id):
        if is_options(i):
            temp = get_options_data(i, begin_t, end_t)
            if len(temp) != 0:
                keys += [i]
                total_theta += [temp['THETA'] *
                                asset_mount[ii] * contract_unit['50ETF']]
    if len(total_theta) == 0:
        return pd.DataFrame()
    else:
        total = pd.concat(total_theta, axis=1, keys=keys)
        total = total.fillna(0)
        return total.sum(axis=1)


def portfolio_volatility(asset_id, asset_mount, cash, begin_t, end_t, time=10):
    total, _ = portfolio_total_value(
        asset_id, asset_mount, cash, begin_t, end_t, time)
    # total=np.log(total)
    res = total.diff(1) / total
    res = res.rolling(time).std()
    return res.dropna()


def portfolio_earning_rate(asset_id, asset_mount, cash, begin_t, end_t, time=10):
    total, _ = portfolio_total_value(
        asset_id, asset_mount, cash, begin_t, end_t, time)
    res = total.diff(time) / total
    return res.dropna()


# 根据组合价值计算取整的期权份数
def cal_option_amt(total_value, option, portion, t1):
    try:
        temp = get_options_data(option, t1, t1)
    except:
        return 0
    if len(temp) <= 0:
        return 0
    else:
        res = total_value * portion / \
            contract_unit['50ETF'] / temp['EXE_PRICE'][-1]
        return int(res + 0.5)


# 根据组合价值计算取整的期货份数
def cal_future_amt(total_value, futures, portion, t1):
    try:
        temp = get_futures_data(futures, t1, t1, 1)
    except:
        return 0
    if len(temp) <= 0:
        return 0
    else:
        res = total_value * portion / \
            contract_unit[futures[:2]] / temp['OPEN'][-1]
        return int(res + 0.5)


# 读取模型训练所需数据
def load_train_data(asset_id, asset_mount, cash, options, begin_t='', end_t='', mode=0, train=0):
    if train == 1:
        data = get_options_data(options, begin_t, end_t, 2)
    else:
        data = get_options_data(options, begin_t, end_t)
    s, _ = portfolio_total_value(
        asset_id, asset_mount, cash, data.index[0], data.index[-1])
    data = pd.concat([data, s], axis=1)
    data.columns = list(data.columns[:-1]) + ['s']
    data = data[~np.isnan(data['s'])]

    data['ds'] = data['s'].diff()
    data['f'] = data['OPEN'] * contract_unit['50ETF']
    data['df'] = data['f'].diff()
    data['real_delta'] = data['df'] / data['ds']
    if mode == 1:
        data['dds'] = data['real_delta'].diff()
        data['real_gamma'] = data['dds'] / data['ds']

    if mode == 1:
        data['gamma_pre'] = data['GAMMA'].shift(-1)
        data['real_gamma'] = data['real_gamma'].shift(-1)

    data['delta_pre'] = data['DELTA'].shift(-1)
    data['real_delta'] = data['real_delta'].shift(-1)  # 错开一天，即为预测下一天

    data['PCT_CHG'] = data['df'] / data['OPEN']
    data['HV_5'] = (data['PCT_CHG'].rolling(window=5).std().values)  # 波动率
    data['HV_10'] = (data['PCT_CHG'].rolling(window=10).std().values)
    data['HV_15'] = (data['PCT_CHG'].rolling(window=15).std().values)
    data['HV_20'] = (data['PCT_CHG'].rolling(window=20).std().values)

    data['SoverK'] = data['s'] / data['EXE_PRICE']
    data['T-t'] = list(map(lambda x: x.days / 365,
                           data['EXE_ENDDATE'] - data.index))

    for i in range(len(data)):  # 隐含波动率没有解的（为空的），用20日波动率替代
        if data['US_IMPLIEDVOL'][i] == 0:
            data['US_IMPLIEDVOL'][i] = data['HV_20'][i].deepcopy()
    if train:
        data = data.dropna()
    else:
        data = data.where(data.notnull(), 0)
    if train:
        # 选取delta在正负0.95以内的数据训练，删去极端值，原因参考论文
        data = data[abs(data['real_delta']) <= 0.95]

    if mode == 1:
        data_train = data[['EXE_PRICE', 's', 'T-t', 'US_IMPLIEDVOL', 'HV_5', 'HV_10', 'HV_15', 'HV_20', 'VOLATILITYRATIO', 'DELTA', 'GAMMA',
                           'VEGA', 'THETA', 'RHO', 'VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE', 'HIGH', 'LOW', 'delta_pre', 'real_delta', 'gamma_pre', 'real_gamma']]
    else:
        data_train = data[['EXE_PRICE', 's', 'T-t', 'US_IMPLIEDVOL', 'HV_5', 'HV_10', 'HV_15', 'HV_20', 'VOLATILITYRATIO', 'DELTA',
                           'GAMMA', 'VEGA', 'THETA', 'RHO', 'VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE', 'HIGH', 'LOW', 'delta_pre', 'real_delta']]
    data_train = data_train.dropna()
    return data_train


# 训练模型
def train_delta_model(protfolio_id, asset_id, asset_mount, cash, options, num=0):
    data_train = load_train_data(asset_id, asset_mount, cash, options, train=1)
    rfr = RandomForestRegressor(n_estimators=15, n_jobs=8)
    rfr.fit(data_train.iloc[:, :-2].values, data_train.iloc[:, -1].values)
    joblib.dump(rfr, str(protfolio_id) + "_delta" + str(num) + ".m")

    return rfr


# 保存模型为生效的模型
def rename_delta_model(protfolio_id):
    try:
        model = joblib.load(str(protfolio_id) + "_delta" + str(0) + ".m")
    except:
        model = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options)
    joblib.dump(model, str(protfolio_id) + "_delta" + str(10) + ".m")
    return

# 重训练更新模型


def retrain_delta_model(protfolio_id, asset_id, asset_mount, cash, options, test=0):
    return train_delta_model(protfolio_id, asset_id, asset_mount, cash, options, num=test * 10)

# 训练模型


def train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options, num):
    data_train = load_train_data(
        asset_id, asset_mount, cash, options, mode=1, train=1)
    rfr = RandomForestRegressor(n_estimators=15, n_jobs=8)
    rfr.fit(data_train.iloc[:, :-4].values, data_train.iloc[:, -1].values)
    joblib.dump(rfr, str(protfolio_id) + "_gamma" + str(num) + ".m")
    return rfr


# 重训练更新模型
def retrain_gamma_model(protfolio_id, asset_id, asset_mount, cash, options1, options2, test=0):
    model1 = train_gamma_model(
        protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    model2 = train_gamma_model(
        protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)
    model3 = train_delta_model(
        protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    model4 = train_delta_model(
        protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)


# 保存模型为生效的模型
def rename_gamma_model(protfolio_id):
    test = 0
    try:
        model1 = joblib.load(str(protfolio_id) +
                             "_gamma" + str(10 * test + 1) + ".m")
    except:
        model1 = train_gamma_model(
            protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    try:
        model2 = joblib.load(str(protfolio_id) +
                             "_gamma" + str(10 * test + 2) + ".m")
    except:
        model2 = train_gamma_model(
            protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)
    try:
        model3 = joblib.load(str(protfolio_id) +
                             "_delta" + str(10 * test + 1) + ".m")
    except:
        model3 = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    try:
        model4 = joblib.load(str(protfolio_id) +
                             "_delta" + str(10 * test + 2) + ".m")
    except:
        model4 = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)
    joblib.dump(model3, str(protfolio_id) + "_delta" + str(11) + ".m")
    joblib.dump(model4, str(protfolio_id) + "_delta" + str(12) + ".m")
    joblib.dump(model1, str(protfolio_id) + "_gamma" + str(11) + ".m")
    joblib.dump(model2, str(protfolio_id) + "_gamma" + str(12) + ".m")
    return

# 模型预测


def fit_delta(protfolio_id, asset_id, asset_mount, cash, options, begin_t, end_t, test=0):
    try:
        model = joblib.load(str(protfolio_id) +
                            "_delta" + str(10 * test) + ".m")
    except:
        model = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options, 10 * test)
    data = load_train_data(asset_id, asset_mount, cash,
                           options, begin_t, end_t, train=0)
    res = model.predict(data.iloc[:, :-2].values)
    res = list(map(lambda x: 0 if x >= 0 else -1 / x, res))
    data['pred'] = res
    return data['pred']


# 模型预测
def fit_gamma(protfolio_id, asset_id, asset_mount, cash, options1, options2, begin_t, end_t, test=0):
    try:
        model1 = joblib.load(str(protfolio_id) +
                             "_gamma" + str(10 * test + 1) + ".m")
    except:
        model1 = train_gamma_model(
            protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    try:
        model2 = joblib.load(str(protfolio_id) +
                             "_gamma" + str(10 * test + 2) + ".m")
    except:
        model2 = train_gamma_model(
            protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)
    try:
        model3 = joblib.load(str(protfolio_id) +
                             "_delta" + str(10 * test + 1) + ".m")
    except:
        model3 = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options1, 10 * test + 1)
    try:
        model4 = joblib.load(str(protfolio_id) +
                             "_delta" + str(10 * test + 2) + ".m")
    except:
        model4 = train_delta_model(
            protfolio_id, asset_id, asset_mount, cash, options2, 10 * test + 2)

    data1 = load_train_data(asset_id, asset_mount, cash,
                            options1, begin_t, end_t, mode=1, train=0)
    data2 = load_train_data(asset_id, asset_mount, cash,
                            options2, begin_t, end_t, mode=1, train=0)
    data = pd.concat([data1, data2], axis=1)
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    gamma1 = model1.predict(data.iloc[:, :21].values)
    gamma2 = model2.predict(data.iloc[:, 25:46].values)
    delta1 = model3.predict(data.iloc[:, :21].values)
    delta2 = model4.predict(data.iloc[:, 25:46].values)
    gamma1 = pd.Series(map(lambda x: 0 if x == 0 else x, gamma1))
    gamma2 = pd.Series(map(lambda x: 0 if x == 0 else x, gamma2))
    delta1 = pd.Series(map(lambda x: 0 if x == 0 else x, delta1))
    delta2 = pd.Series(map(lambda x: 0 if x == 0 else x, delta2))
    temp = gamma1 * delta2 - gamma2 * delta1
    x1 = gamma2 / temp
    x2 = gamma1 / (-temp)
    x1 = x1.map(lambda x: 0 if x <= 0 else x)
    x2 = x2.map(lambda x: 0 if x <= 0 else x)
    data['x1'] = list(x1)
    data['x2'] = list(x2)
    return data[['x1', 'x2']]


# 读取模型训练所需数据
def load_train_future_data(asset_id, asset_mount, cash, future, begin_t='', end_t=''):
    data = get_futures_data(future, begin_t, end_t)['OPEN']
    s, _ = portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    data = pd.concat([data, s], axis=1, keys=['future', 's'])
    data['r_f'] = data['future'].diff() / data['future']
    data['r_s'] = data['s'].diff() / data['s']

    data = data[~np.isnan(data['s'])]
    data = data.dropna()
    return data[['r_f', 'r_s']]


# 重训练更新模型
def retrain_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num=0):
    return train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num=num)


# 保存模型为生效的模型
def rename_beta_model(protfolio_id):
    try:
        model = joblib.load(str(protfolio_id) + "_beta" + str(0) + ".m")
    except:
        model = train_beta_model(
            protfolio_id, asset_id, asset_mount, cash, futures, 0)
    joblib.dump(model, str(protfolio_id) + "_beta" + str(10) + ".m")
    return


# 训练模型
def train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures, num=0):
    data_train = load_train_future_data(asset_id, asset_mount, cash, futures)
    model = linear_model.LinearRegression()
    model.fit(data_train.iloc[:, 0:1].values, data_train.iloc[:, 1].values)
    joblib.dump(model, str(protfolio_id) + "_beta" + str(num) + ".m")
    return model


# 模型预测
def fit_beta(protfolio_id, asset_id, asset_mount, cash, futures, test=0):
    try:
        model = joblib.load(str(protfolio_id) + "_beta" +
                            str(test * 10) + ".m")
    except:
        model = train_beta_model(
            protfolio_id, asset_id, asset_mount, cash, futures, test * 10)
    res = model.coef_
    return res[0]

# 计算涨跌幅


def cal_option_change_rate(option, t):
    dat = get_options_data(option, t, t)
    res = (dat['CLOSE'] - dat['PRECLO']) / dat['PRECLO']
    return res[0]


def cal_future_change_rate(future, t):
    dat = get_futures_data(future, t, t)
    res = (dat['CLOSE'] - dat['PRECLO']) / dat['PRECLO']
    return res[0]

# 生成推荐期权


def generate_recommend_option_delta(protfolio_id, asset_id, asset_mount, cash):
    sql = "select * from OPTIONINFO "
    sql_dat = pd.DataFrame(list(c.execute(sql)), columns=[
                           'index', 'TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'FIRST_DATE', 'LAST_DATE'])
    sql = "select DATE from " + \
        sql_dat['TRADECODE'][0][-2:] + sql_dat['TRADECODE'][0][:8]
    today = list(c.execute(sql))[-1][0]
    data = []
    for ii, i in enumerate(sql_dat['TRADECODE']):
        sql = 'select OPEN,HIGH,LOW,CLOSE,OI,VOLUME,IMPLIEDVOL,VOLATILITYRATIO from ' + \
            i[-2:] + i[:8] + " where DATE==\'" + today + "\'"
        data += [list(list(c.execute(sql))[0])]
    sql_dat = pd.concat([sql_dat, pd.DataFrame(data)], axis=1)
    sql_dat.columns = ['index', 'TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'FIRST_DATE', 'LAST_DATE'] + \
        ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'OI',
            'VOLUME', 'IMPLIEDVOL', 'VOLATILITYRATIO']
    today = pd.Timestamp(today)
    sql_dat['chg'] = sql_dat['TRADECODE'].map(
        lambda x: cal_option_change_rate(x, str(today)[:10]))
    sql_dat['LAST_DATE'] = sql_dat['LAST_DATE'].map(pd.Timestamp)
    sql_dat['days_left'] = sql_dat['LAST_DATE'] - today
    sql_dat['days_left'] = sql_dat['days_left'].map(lambda x: int(x.days))
    sql_dat = sql_dat[np.array(sql_dat['days_left'] >= 40) & np.array(
        sql_dat['days_left'] <= 120)]
    sql_selected = sql_dat[np.array(sql_dat['EXE_PRICE'] >= 2.1) & np.array(
        sql_dat['EXE_PRICE'] <= 2.9)]
    sql_selected = sql_selected[sql_selected['EXE_MODE'] == '认购']
    sql_selected = sql_selected.sort_values(by='chg', ascending=False)
    sql_final = pd.concat(
        [sql_selected, sql_dat[[x not in sql_selected.index for x in sql_dat.index]]])
    return sql_final[['TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'OI', 'VOLUME', 'IMPLIEDVOL', 'VOLATILITYRATIO', 'chg', 'days_left']]


def generate_recommend_option_gamma(protfolio_id, asset_id, asset_mount, cash):
    sql = "select * from OPTIONINFO "
    sql_dat = pd.DataFrame(list(c.execute(sql)), columns=[
                           'index', 'TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'FIRST_DATE', 'LAST_DATE'])
    sql = "select DATE from " + \
        sql_dat['TRADECODE'][0][-2:] + sql_dat['TRADECODE'][0][:8]
    today = list(c.execute(sql))[-1][0]
    data = []
    for ii, i in enumerate(sql_dat['TRADECODE']):
        sql = 'select OPEN,HIGH,LOW,CLOSE,OI,VOLUME,IMPLIEDVOL,VOLATILITYRATIO from ' + \
            i[-2:] + i[:8] + " where DATE==\'" + today + "\'"
        data += [list(list(c.execute(sql))[0])]
    sql_dat = pd.concat([sql_dat, pd.DataFrame(data)], axis=1)
    sql_dat.columns = ['index', 'TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'FIRST_DATE', 'LAST_DATE'] + \
        ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'OI',
            'VOLUME', 'IMPLIEDVOL', 'VOLATILITYRATIO']
    today = pd.Timestamp(today)
    sql_dat['chg'] = sql_dat['TRADECODE'].map(
        lambda x: cal_option_change_rate(x, str(today)[:10]))
    sql_dat['LAST_DATE'] = sql_dat['LAST_DATE'].map(pd.Timestamp)
    sql_dat['days_left'] = sql_dat['LAST_DATE'] - today
    sql_dat['days_left'] = sql_dat['days_left'].map(lambda x: int(x.days))
    sql_dat = sql_dat[np.array(sql_dat['days_left'] >= 40) & np.array(
        sql_dat['days_left'] <= 120)]
    sql_selected = sql_dat[np.array(sql_dat['EXE_PRICE'] >= 2.3) & np.array(
        sql_dat['EXE_PRICE'] <= 2.7)]
    sql_selected = sql_selected.sort_values(by='chg', ascending=False)
    sql_final = pd.concat(
        [sql_selected, sql_dat[[x not in sql_selected.index for x in sql_dat.index]]])
    return sql_final[['TRADECODE', 'EXE_PRICE', 'EXE_MODE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'OI', 'VOLUME', 'IMPLIEDVOL', 'VOLATILITYRATIO', 'chg', 'days_left']]

# 生成推荐期货


def generate_recommend_future(protfolio_id, asset_id, asset_mount, cash):
    sql = "select * from FUTUREINFO "
    sql_dat = pd.DataFrame(list(c.execute(sql)), columns=[
                           'index', 'TRADECODE', 'WIN_CODE', 'NAME', 'FIRST_DATE', 'LAST_DATE'])
    sql = "select DATE from " + sql_dat['TRADECODE'][0]
    today = list(c.execute(sql))[-1][0]
    data = []
    for ii, i in enumerate(sql_dat['TRADECODE']):
        sql = 'select OPEN,HIGH,LOW,CLOSE,SETTLE,OI,VOLUME,AMT,VR from ' + \
            i + " where DATE==\'" + today + "\'"
        data += [list(list(c.execute(sql))[0])]
    sql_dat = pd.concat([sql_dat, pd.DataFrame(data)], axis=1)
    sql_dat.columns = ['index', 'TRADECODE', 'WIN_CODE', 'NAME', 'FIRST_DATE',
                       'LAST_DATE'] + ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE', 'OI', 'VOLUME', 'AMT', 'VR']
    today = pd.Timestamp(today)
    sql_dat['chg'] = sql_dat['TRADECODE'].map(
        lambda x: cal_future_change_rate(x, str(today)[:10]))
    sql_dat['LAST_DATE'] = sql_dat['LAST_DATE'].map(pd.Timestamp)
    sql_dat['days_left'] = sql_dat['LAST_DATE'] - today
    sql_dat['days_left'] = sql_dat['days_left'].map(lambda x: int(x.days))
    sql_selected = sql_dat[np.array(sql_dat['days_left'] >= 30) & np.array(
        sql_dat['days_left'] <= 150)]
    sql_selected = sql_selected.sort_values(by='chg', ascending=False)
    sql_final = pd.concat(
        [sql_selected, sql_dat[[x not in sql_selected.index for x in sql_dat.index]]])
    return sql_dat[['TRADECODE', 'NAME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE', 'OI', 'VOLUME', 'AMT', 'VR', 'chg', 'days_left']]


# 计算组合beta值
def get_portfolio_beta(asset_id, weight_list):
    from sklearn.linear_model import LinearRegression
    stock_list = [id[-2:] + id[:6] for id in asset_id]
    beta_list = []
    for each in stock_list:
        df_total = pd.DataFrame(list(c.execute("SELECT * from " + each)))
        if df_total.empty:
            return np.nan
        r = df_total.iloc[:, 8]  # 股票收益率
        df_total2 = pd.DataFrame(list(c.execute("SELECT * from MARKET")))
        rf = df_total2.iloc[:, 2]  # 无风险收益率
        rm = df_total2.iloc[:, 3]  # 市场收益率
        r_excess = r - rf
        rm_excess = r - rm
        r_ewma = r_excess.ewm(halflife=63).mean()
        rm_ewma = rm_excess.ewm(halflife=63).mean()
        r_ewma = np.array([r_ewma.tolist()]).T
        rm_ewma = np.array([rm_ewma.tolist()]).T
        mx = LinearRegression()
        mx.fit(rm_ewma, r_ewma)
        beta = mx.coef_[0][0]
        beta_list = beta_list + [beta]
    beta_mean = 0
    for index in range(0, len(weight_list)):
        beta_mean = beta_mean + beta_list[index] * weight_list[index]
    return beta_mean
