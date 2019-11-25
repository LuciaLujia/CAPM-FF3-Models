#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/11/23 11:25 
# @Author : Lujia (Lucia) Huang
# @Contact : huang.lujia@outlook.com
# @File : FF3.py
# @Version :
# @Description: 对沪深两市房地产板块股票的FF3实证研究，
#               年份: 'year_2016''year_2017''year_2018'
#               对股票进行两种分组法：
#                   1）'name_4_groups': 分成SL/SH/BL/BH四组
#                   2）'name_6_groups': 分成SL/SM/SH/BL/BM/BH六组（稳健性检验）


import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import re


#获取需要的数据
data_rmrf = pd.read_excel('data&program/data_FF3.xlsx', sheet_name="股指", header=0, index_col=0)
data_ri = pd.read_excel('data&program/data_FF3.xlsx', sheet_name="收盘价", header=0, index_col=0)
data_value = pd.read_excel('data&program/data_FF3.xlsx', sheet_name="总市值", header=0, index_col=0)
data_PB = pd.read_excel('data&program/data_FF3.xlsx', sheet_name="市净率倒数", header=0, index_col=0)


#共有133支股票
stk_num = 133


#由于需要按年度对股票进行分组，用字典管理每年的各类参数
year_dic = {
    'year_2016': {'week_range': range(0, 49), 'week_range_from_0': range(0, 49), 'last_day': '2016-12-30'},
    'year_2017': {'week_range': range(49, 100), 'week_range_from_0': range(0, 51), 'last_day': '2017-12-29'},
    'year_2018': {'week_range': range(100, 151), 'week_range_from_0': range(0, 51), 'last_day': '2018-12-28'}
}


#无风险收益率rf
Rft = data_rmrf.loc[:, "定期存款3M"].shift(1) / 52
Rft = list(Rft.dropna())


#所有股票的周收益率rit
Rit_dic = {}
for i in range(0, stk_num):
    stkname = data_ri.columns[i]
    Rit = (data_ri.iloc[:, i] / data_ri.iloc[:, i].shift(1)) - 1
    Rit = list(Rit.dropna())
    Rit_dic[stkname] = Rit
Rit_table = pd.DataFrame(Rit_dic)


def name_groups(year):
    '''
    对不同年份的股票，根据期末市值和账面市值比（市净率的倒数）进行分组
    :param year: 进行分组的年份，可选'year_2016''year_2017''year_2018'
    :return: 两种分组法共10组(4+6)股票名称，储存在字典里
    '''
    def group_stk(data, year_last_day, start_pct, end_pct, num):
        '''
        股票分组函数
        :param data: 针对分组的数据集，可选期末市值数据(data_value)或期末账面市值比(data_PB)
        :param year_last_day: 参与分组的年度最后一天，可从year_dic里选取各年度的'last_day'参数
        :param start_pct: 分组起始百分比位置(lower,asc)
        :param end_pct: 分组结束百分比位置(higher,asc)
        :param num: 参与分组的股票数目
        :return: 该分组的股票名称
        '''
        data = data.sort_values(by=year_last_day)
        stkname = list(data.iloc[round(start_pct*num):round(end_pct*num)].index)
        return stkname
    Sname = group_stk(data_value, year_dic[year]['last_day'], 0, 0.5, stk_num)
    Bname = group_stk(data_value, year_dic[year]['last_day'], 0.5, 1, stk_num)
    SLname_4 = group_stk(data_PB.loc[Sname], year_dic[year]['last_day'], 0, 0.5, len(Sname))
    SHname_4 = group_stk(data_PB.loc[Sname], year_dic[year]['last_day'], 0.5, 1, len(Sname))
    BLname_4 = group_stk(data_PB.loc[Bname], year_dic[year]['last_day'], 0, 0.5, len(Bname))
    BHname_4 = group_stk(data_PB.loc[Bname], year_dic[year]['last_day'], 0.5, 1, len(Bname))
    SLname_6 = group_stk(data_PB.loc[Sname], year_dic[year]['last_day'], 0, 0.33, len(Sname))
    SMname_6 = group_stk(data_PB.loc[Sname], year_dic[year]['last_day'], 0.33, 0.67, len(Sname))
    SHname_6 = group_stk(data_PB.loc[Sname], year_dic[year]['last_day'], 0.67, 1, len(Sname))
    BLname_6 = group_stk(data_PB.loc[Bname], year_dic[year]['last_day'], 0, 0.33, len(Bname))
    BMname_6 = group_stk(data_PB.loc[Bname], year_dic[year]['last_day'], 0.33, 0.67, len(Bname))
    BHname_6 = group_stk(data_PB.loc[Bname], year_dic[year]['last_day'], 0.67, 1, len(Bname))
    group_dic = {
        'name_4_groups':{'SL':SLname_4,'SH':SHname_4,'BL':BLname_4,'BH':BHname_4},
        'name_6_groups':{'SL':SLname_6,'SM':SMname_6,'SH':SHname_6,'BL':BLname_6,'BM':BMname_6,'BH':BHname_6}
    }
    return group_dic


def weighted_zit_zmt(group_type,group_name):
    '''
    求两种分组方式的每组股票的组合收益率Zit(=E[rit]-rf)和市场收益率Zmt(=E[rmt]-rf)
    按年度计算，再把三年的结果拼接
    :param group_type: 分组方法，可选'name_4_groups''name_6_groups'
    :param group_name: 股票分组，可选'SL''SH''SM''BL''BH''BM'
    :return: 该分组方式下的该组，其三年的组合收益率Zit和市场收益率Zmt
    '''
    p_SH = re.compile(r'SH')
    p_SZ = re.compile(r'SZ')
    Zit = []
    Zmt = []
    for year in ['year_2016','year_2017','year_2018']:
        Zmt_by_year = []
        Zit_by_year = []
        # 加权市场超额收益率（按股票数量分配上证和深成的权重）
        wsh = (len(p_SH.findall(str(name_groups(year)[group_type][group_name])))
               / len(name_groups(year)[group_type][group_name]))
        wsz = (len(p_SZ.findall(str(name_groups(year)[group_type][group_name])))
               / len(name_groups(year)[group_type][group_name]))
        Rmt = (((data_rmrf.loc[:, "深证成指399001.SZ"] / data_rmrf.loc[:, "深证成指399001.SZ"].shift(1)) - 1) * wsz
               + ((data_rmrf.loc[:, "上证综指000001.SH"] / data_rmrf.loc[:, "上证综指000001.SH"].shift(1)) - 1) * wsh)
        Rmt = list(Rmt.dropna())
        Zmt_by_year = [Rmt[i] - Rft[i] for i in year_dic[year]['week_range']]
        # 加权组合收益率（按股票期末市值分配个股收益率的权重）
        weights = data_value.loc[name_groups(year)[group_type][group_name]].apply(lambda x: x / x.sum(), axis=0)
        for week in year_dic[year]['week_range']:
            WRit = 0
            for name in name_groups(year)[group_type][group_name]:
                w = weights.loc[name,year_dic[year]['last_day']]
                WRit = WRit + Rit_table.loc[week,name]*w
            Zit_by_year += [WRit]
        # 拼接三年的结果
        Zit += Zit_by_year
        Zmt += Zmt_by_year
    # 加权组合超额收益率（减去rf）
    Zit = [Zit[i] - Rft[i] for i in range(len(Zit))]
    return Zit,Zmt


def smb_hml():
    '''
    对两种分组方法，分别计算SMB和HML，并把结果储存在字典里
    :return: 字典
    '''
    smb_hml_dic = {'name_4_groups':{},'name_6_groups':{}}
    def weighted_zit_by_year(year, name_group):
        '''
        求该年、该分组方式下的组合超额收益率Zit
        :param year: 概念
        :param name_group: 该年、该分组下的所有股票名称
        :return: Zit
        '''
        Zit_by_year = []
        weights = data_value.loc[name_group].apply(lambda x: x / x.sum(), axis=0)
        for week in year_dic[year]['week_range']:
            WRit = 0
            for name in name_group:
                w = weights.loc[name, year_dic[year]['last_day']]
                WRit = WRit + Rit_table.loc[week, name] * w
            Zit_by_year += [WRit]
        Zit_by_year = [Zit_by_year[i] - Rft[i] for i in range(len(Zit_by_year))]
        return Zit_by_year
    # 判断是哪种分组方式，分别计算SMB和HML
    # 'name_4_groups': SMB = (SL+SH)/2 - (BL+BH)/2
    #                  HML = (BH+SH)/2 - (BL+SL)/2
    # 'name_6_groups': SMB = (SL+SH+SM)/3 - (BL+BH+BM)/3
    #                  HML = (BH+SH)/2 - (BL+SL)/2
    for group_type in ('name_4_groups','name_6_groups'):
        SMB = []
        HML = []
        if group_type == 'name_4_groups':
            for year in ['year_2016','year_2017','year_2018']:
                SMB += [(
                        (weighted_zit_by_year(year,name_groups(year)['name_4_groups']['SL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_4_groups']['SH'])[i])/2
                        -
                        (weighted_zit_by_year(year,name_groups(year)['name_4_groups']['BL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_4_groups']['BH'])[i])/2
                )for i in year_dic[year]['week_range_from_0']]
                HML += [(
                        (weighted_zit_by_year(year,name_groups(year)['name_4_groups']['BH'])[i]+
                        weighted_zit_by_year(year,name_groups(year)['name_4_groups']['SH'])[i])/2
                        -
                        (weighted_zit_by_year(year,name_groups(year)['name_4_groups']['BL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_4_groups']['SL'])[i])/2
                )for i in year_dic[year]['week_range_from_0']]
        elif group_type == 'name_6_groups':
            for year in ['year_2016','year_2017','year_2018']:
                SMB += [(
                        (weighted_zit_by_year(year,name_groups(year)['name_6_groups']['SL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['SM'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['SH'])[i])/3
                        -
                        (weighted_zit_by_year(year,name_groups(year)['name_6_groups']['BL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['BH'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['BM'])[i])/3
                )for i in year_dic[year]['week_range_from_0']]
                HML += [(
                        (weighted_zit_by_year(year,name_groups(year)['name_6_groups']['BH'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['SH'])[i])/2
                        -
                        (weighted_zit_by_year(year,name_groups(year)['name_6_groups']['BL'])[i]+
                         weighted_zit_by_year(year,name_groups(year)['name_6_groups']['SL'])[i])/2
                )for i in year_dic[year]['week_range_from_0']]
        smb_hml_dic[group_type]['SMB'] = SMB
        smb_hml_dic[group_type]['HML'] = HML
    return smb_hml_dic


def ols(group_type,group_name):
    '''
    回归
    :param group_type: 分组方法，可选'name_4_groups''name_6_groups'
    :param group_name: 股票分组，可选'SL''SH''SM''BL''BH''BM'
    :return: 回归结果
    '''
    Y = weighted_zit_zmt(group_type,group_name)[0]
    X_addcons = sm.add_constant(np.column_stack((weighted_zit_zmt(group_type,group_name)[1],
                                                smb_hml_dic[group_type]['SMB'],
                                                smb_hml_dic[group_type]['HML'])))
    model_ff3 = sm.OLS(endog=Y, exog=X_addcons).fit()
    return model_ff3.summary()


def output():
    '''
    对两种分组方法下的各个分组跑回归，把结果保存在csv文件里
    :return: csv
    '''
    type_name_dic = {
        'name_4_groups':['SL','SH','BL','BH'],
        'name_6_groups':['SL','SM','SH','BL','BM','BH']
    }
    w = open(f"FF3_results{datetime.datetime.now().strftime('%Y-%m-%d %H@%M@%S')}.csv", 'w+')
    for group_type in ('name_4_groups','name_6_groups'):
        for group_name in type_name_dic[group_type]:
            w.write('{type}_{name}'.format(type = group_type,name = group_name))
            w.write('\n\n')
            w.write(ols(group_type, group_name).as_csv())
            w.write('\n\n\n\n')
    w.close()

#生成两种分组方法的SMB和HML字典
smb_hml_dic = smb_hml()

#导出结果csv
output()
