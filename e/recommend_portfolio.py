import sqlite3
import pandas as pd
import numpy as np
from scipy import optimize
import math
import random

## 设定总股票数
total_stock = 99
## 风格因子数目
factor_num = 7
## 寻找组合迭代最大次数
MAX_ITER_NUM = 100


# %% 从STOCKINFO到STOCK的表名称转换
def trans_name(info_code):
    '''
        从STOCKINFO到STOCK的表名称转换
    :param info_code: str
        0000001.SZ
    :return: str
        SZ000001
    '''
    letter_num = info_code.split('.')
    return letter_num[1] + letter_num[0]


# %% （内部函数）获得总股票收益率池
def get_code_gain(days=200, count=99):
    '''
    计算股票收益率（未排序）
    :param days: int
    :param count: int
    :return: df
        第一列 stockcode 第二列 returns
        共 count（默认200）支股票，returns 求的是days（默认100）天的收益率
    '''
    conn = sqlite3.connect('data/fin_set.db')
    c = conn.cursor()  # 创建游标

    code_df = pd.DataFrame(
        list(c.execute("SELECT TRADECODE from " + 'STOCKINFO'))
    )
    exist_code_ls = np.array(code_df.iloc[:count, :].T)[0]  # 提取前count = 100条股票名称
    gain_ls = []
    stockcode_ls = []
    for info_code in exist_code_ls:
        if info_code == '000029.SZ':
            print('29跳过')
            continue
        table_code = trans_name(info_code)
        stockcode_ls.append(table_code)
        chg_df = pd.DataFrame(list(c.execute("SELECT CLOSE from " + table_code)))
        this_gain = chg_df.iloc[days, 0] - chg_df.iloc[0, 0]
        gain_ls.append(float(this_gain))

    code_gain = pd.DataFrame({'stockcode': stockcode_ls, 'returns': gain_ls})
    return code_gain


# %% （内部函数）获得按收益率排序的总股票池
def get_sorted_code_gain(days=200, count=99):
    '''
    获得按收益率排序的总股票池
    :param days: int
        选取几日收益率，default 200
    :param count: int
        总股票数, default 100
    :return: df
        第一列 stockcode 第二列 returns
    '''
    code_gain = get_code_gain(days, count)
    sorted_cg = code_gain.sort_values(by='returns', ascending=False)
    return sorted_cg


# %% 获得符合收益率档位的股票池
def get_targeted_code_gain(return_level='low'):
    '''
    获得符合收益率要求的股票池
    :param return_level: str
        收益率等级 'low', 'high','mid'
    :return: dataframe
        第一列 stockcode 第二列 returns
    '''
    sorted_cg = get_sorted_code_gain(count=total_stock)
    count = len(sorted_cg)
    if return_level == 'low':
        return sorted_cg.iloc[-int(count / 3):, :]
    elif return_level == 'high':
        return sorted_cg.iloc[:int(count / 3), :]
    else:
        return sorted_cg.iloc[int(count / 3): 2 * int(count / 3), :]


# %% 最优化问题求解器
def weight_optimizor(stock_gain_df, beta_df, beta_threshold):
    '''
    optimizor
    :param stock_gain_df: df
        组合中的 股票代码+收益率
    :param beta_df: df
        约束条件系数矩阵n*m，每列是一只股票（共n值股票），每行是一个因子值（共m个因子）eg.
        pd.DataFrame({'stock1':[1,2,3],'stock2':[2,3,4]})
    :param beta_threshold: 一维 ndarray
        限额系数， beta阈值向量
        np.array([2, 3, 1])
    :return: df
        两列： stockcode个股代码  weight个股权重
    '''
    c = stock_gain_df.iloc[:, 1].values  # 目标函数系数
    A_ub = beta_df.values  # 系数矩阵，m * n
    n, m = A_ub.shape[1], A_ub.shape[0]  # 总股票数n,总因子数m

    A_eq = np.array([np.zeros(n) + 1])
    b_eq = np.array([1])
    bounds = (0, 1)

    solve = optimize.linprog(
        c=-c, A_ub=A_ub, b_ub=beta_threshold,
        A_eq=A_eq, b_eq=b_eq, bounds=bounds
    )
    return solve


# %% 获取约束矩阵（beta系数矩阵）
def get_beta_df(stockcode_ls):
    '''
    获取约束矩阵（beta系数矩阵）
    :param list
    :return df, 每列是一只股票（共 n 值股票），每行是一个因子值（共 m个因子）eg.
        pd.DataFrame({'stock1':[1,2,3],'stock2':[2,3,4]})
    '''
    conn = sqlite3.connect('data/fin_set.db')
    c = conn.cursor()  # 创建游标

    code_beta = pd.DataFrame(
        list(c.execute("SELECT STOCKCODE,B1,B2,B3,B4,B5,B6,B7 from " + 'BETA')),
        columns=['STOCKCODE', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )

    name_ls = code_beta.iloc[:, 0]
    code_ls = []
    for info_code in name_ls:
        table_code = trans_name(info_code)
        code_ls.append(table_code)
    code_beta['STOCKCODE'] = code_ls

    filtered_df = code_beta[code_beta['STOCKCODE'].isin(stockcode_ls)].T

    filtered_df.columns = filtered_df.iloc[0, :]

    filtered_df = filtered_df.iloc[1:, :]
    return filtered_df


# %% 获取限额向量（beta阈值ndarray）调用此API的时候要给出！！！！
def get_beta_threshold(beta_threshold=np.full(factor_num, 0.4)):
    '''
    获取限额向量（beta阈值ndarray）
    :param beta_threshold: ndarray
        beta阈值向量，默认参数 np.full(factor_num,0.4)
    :return: m个因子的beta阈值, 一维 ndarray，长度 m
        np.array([2, 3, 1])
    '''
    return beta_threshold


# %%
target_cg = get_targeted_code_gain(return_level='low').reset_index(drop=True)
stock_num = 10
total = len(target_cg)
# a  = [random.randint(0,total-1) for _ in range(stock_num)]
a = random.sample(range(0, total - 1), stock_num)
print(a)

stock_gain_df = target_cg.iloc[a, :]
stockcode_ls = np.array(stock_gain_df['stockcode']).tolist()

conn = sqlite3.connect('data/fin_set.db')
c = conn.cursor()  # 创建游标

code_beta = pd.DataFrame(
    list(c.execute("SELECT STOCKCODE,B1,B2,B3,B4,B5,B6,B7 from " + 'BETA')),
    columns=['STOCKCODE', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
)

name_ls = code_beta.iloc[:, 0]
code_ls = []
for info_code in name_ls:
    table_code = trans_name(info_code)
    code_ls.append(table_code)
code_beta['STOCKCODE'] = code_ls

filtered_df = code_beta[code_beta['STOCKCODE'].isin(stockcode_ls)].T

filtered_df.columns = filtered_df.iloc[0, :]

filtered_df = filtered_df.iloc[1:, :]



## 组合中股票数k（默认是10），随机从该收益率档位（默认是收益水平low)的股票池中挑选k个股票
## 对这k个股票权重进行最优化求解，获得推荐
def recommend_portfolio(beta_threshold=np.full(factor_num, 0.4), stock_num=10, return_level='low'):
    '''
    组合中股票数k（默认是10），随机从该收益率档位（默认是收益水平low)的股票池中挑选k个股票
    对这k个股票权重进行最优化求解，获得推荐
    # 说明： 均为可选参数，找不到合适解或个股数目要求太多时会提示错误类型
    # 提醒：当return_level设为'high'时，阈值最好不低于0.45，否则可能找不到合适解
    :param beta_threshold: ndarray
        beta阈值向量，默认参数 np.full(factor_num,0.4)
    :param stock_num: int
        组合中的个股种类数
    :param return_level: str
        收益率档位，'low','mid','high'三种
    :return: df
        两列 stockcode weight
        （总 weight是1）
    '''
    if stock_num > 32:
        print('Too many stocks in the portfolio. ')
        return

    target_cg = get_targeted_code_gain(return_level=return_level).reset_index(drop=True)
    total = len(target_cg)
    random.seed(17)
    success = False
    iter_num = 0
    while success is False and iter_num < MAX_ITER_NUM:

        # 随机在该收益率档位（默认是收益水平low)的股票池取stock_num支（默认10支）股票+收益率
        # 存为stock_gain_df
        a = random.sample(range(0, total - 1), stock_num)
        stock_gain_df = target_cg.iloc[a, :]

        # 获取约束矩阵（beta系数）
        beta_df = get_beta_df(np.array(stock_gain_df['stockcode']).tolist())
        # 获取限额向量（阈值beta)
        beta_threshold = get_beta_threshold(beta_threshold)
        opt = weight_optimizor(stock_gain_df, beta_df, beta_threshold)

        if opt.success is True:
            success = True
            return pd.DataFrame({'stockcode': beta_df.columns, 'weight': opt.x})
        else:
            iter_num = iter_num + 1
            continue

    if success is False or iter_num == MAX_ITER_NUM:
        print('Failed to find a portfolio in our pool within ' + str(
            MAX_ITER_NUM) + 'iterations,Please adjust your requirements.')
        return False


# %%================ 此api调用示例 =======================

#rec_portfolio = recommend_portfolio(beta_threshold=np.full(factor_num, 0.45), stock_num=30, return_level='mid')
# 说明： 均为可选参数，找不到合适解或个股数目要求太多时会提示错误类型
# 提醒：当return_level设为'high'时，阈值最好不低于0.45，否则可能找不到合适解
# beta_threshold 默认np.full(factor_num,0.4)
# stock_num 默认10，不超过32
# return_level 默认'low'
#print(rec_portfolio)


