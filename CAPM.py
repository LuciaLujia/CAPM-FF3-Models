#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/11/23 13:53 
# @Author : Lujia (Lucia) Huang
# @Contact : huang.lujia@outlook.com
# @File : CAPM.py
# @Version :
# @Description : 对沪深两市地产板块股票的CAPM实证研究


import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


data_new = pd.read_excel('data_CAPM.xlsx',sheet_name="new", header=0, index_col=0)


'''
===========================================Run by Single Stocks=================================================
'''

def single_sh_sz_func(data):
    '''
    对深交所和上交所的房地产股票分别跑CAPM模型，Rmt使用深成指和上证指的加权平均
    :param data: 使用的数据集名称，当前为“data_new”
    :return: 每只股票CAPM模型的β，Rsquare和Pvalue
    '''
    beta = []
    rsquared = []
    pvalue = []
    name = []
    # 分别计算SH指数和SZ指数从t到t-1的Rmt，再加权平均（根据股票数量，73只上海股，60只深圳股）
    Rmt = (np.log(data.iloc[:,1]/data.iloc[:,1].shift(1))*(73/133)
         + np.log(data.iloc[:,2]/data.iloc[:,2].shift(1))*(60/133))
    Rmt = list(Rmt.dropna())
    # 生成Rmt自变量，以及eit常数项（=1）
    Rmt_addcons = sm.add_constant(Rmt)
    for i in range(3,136):
        # 生成每一只股票在t到t-1的Rit收益率
        Rit = data.iloc[:,i]/data.iloc[:,i].shift(1)
        Rit = list(Rit.dropna())
        # 对每一只股票跑OLS回归
        model_single_sh_sz = sm.OLS(endog=Rit,exog=Rmt_addcons).fit()
        beta += [model_single_sh_sz.params[1].item()]
        rsquared += [model_single_sh_sz.rsquared.item()]
        pvalue += [format(float(model_single_sh_sz.pvalues[1]), '.6f')]
        name += [data.columns[i]]
    model_single_sh_sz_result_dic = {"name":name,"beta":beta,"rsquared":rsquared,"pvalue":pvalue}
    model_single_sh_sz_result_table = pd.DataFrame(model_single_sh_sz_result_dic)
    return model_single_sh_sz_result_table

def single_sh_func(data):
    '''
    对上交所的房地产股票分别跑CAPM模型，Rmt使用上证指
    :param data: 使用的数据集名称，当前为“data_new”
    :return: 每只股票CAPM模型的β，Rsquare和Pvalue
    '''
    beta = []
    rsquared = []
    pvalue = []
    name = []
    Rmt = np.log(data.iloc[:,1]/data.iloc[:,1].shift(1))
    Rmt = list(Rmt.dropna())
    Rmt_addcons = sm.add_constant(Rmt)
    for i in range(63,136):
        Rit = data.iloc[:,i]/data.iloc[:,i].shift(1)
        Rit = list(Rit.dropna())
        model_single_sh = sm.OLS(endog=Rit,exog=Rmt_addcons).fit()
        beta += [model_single_sh.params[1].item()]
        rsquared += [model_single_sh.rsquared.item()]
        pvalue += [format(float(model_single_sh.pvalues[1]), '.6f')]
        name += [data.columns[i]]
    model_single_sh_result_dic = {"name":name,"beta":beta,"rsquared":rsquared,"pvalue":pvalue}
    model_single_sh_result_table = pd.DataFrame(model_single_sh_result_dic)
    return model_single_sh_result_table

def single_sz_func(data):
    '''
    对深交所的房地产股票分别跑CAPM模型，Rmt使用深成指
    :param data: 使用的数据集名称，当前为“data_new”
    :return: 每只股票CAPM模型的β，Rsquare和Pvalue
    '''
    beta = []
    rsquared = []
    pvalue = []
    name = []
    Rmt = np.log(data.iloc[:,2]/data.iloc[:,2].shift(1))*(60/133)
    Rmt = list(Rmt.dropna())
    Rmt_addcons = sm.add_constant(Rmt)
    for i in range(3,63):
        Rit = data.iloc[:,i]/data.iloc[:,i].shift(1)
        Rit = list(Rit.dropna())
        model_single_sz = sm.OLS(endog=Rit,exog=Rmt_addcons).fit()
        beta += [model_single_sz.params[1].item()]
        rsquared += [model_single_sz.rsquared.item()]
        pvalue += [format(float(model_single_sz.pvalues[1]), '.6f')]
        name += [data.columns[i]]
    model_single_sz_result_dic = {"name":name,"beta":beta,"rsquared":rsquared,"pvalue":pvalue}
    model_single_sz_result_table = pd.DataFrame(model_single_sz_result_dic)
    return model_single_sz_result_table

def output_single():
    '''
    输出结果在excel里
    :return: excel
    '''
    writer=pd.ExcelWriter(f"CAPM_result_by_stocks_{datetime.datetime.now().strftime('%Y-%m-%d %H@%M@%S')}.xlsx")
    single_sh_sz_func(data_new).to_excel(writer, sheet_name="%s" % "sh_sz")
    single_sh_func(data_new).to_excel(writer, sheet_name="%s" % "sh")
    single_sz_func(data_new).to_excel(writer, sheet_name="%s" % "sz")
    writer.save()

output_single()


'''
======================================Run by the Whole Real Estate Sector==============================================
'''

def port_sh_sz_func(data):
    '''
    深交所和上交所的所有房地产股票形成投资组合，Rpt为简单算术平均
    :param data:使用的数据集名称，当前为“data_new”
    :return:OLS回归结果
    '''
    Rit_dic = {}
    Rpt = []
    # 73只上海股，60只深圳股加权得Rmt
    Rmt = (np.log(data.iloc[:,1]/data.iloc[:,1].shift(1))*(73/133)
         + np.log(data.iloc[:,2]/data.iloc[:,2].shift(1))*(60/133))
    Rmt = list(Rmt.dropna())
    # 无风险利率Rf是三个月定期利率，按单利计算方式折合到每周
    Rft = list(data.iloc[1:,0]/52)
    for i in range(3,136):
        # 生成每一只股票在t到t-1的Rit收益率
        Rit = data.iloc[:, i]/data.iloc[:, i].shift(1)
        Rit = list(Rit.dropna())
        Rit_dic[i] = Rit
    Rit_table = pd.DataFrame(Rit_dic)
    # 算所有股票收益率的平均值，得组合收益率Rpt
    Rpt = Rit_table.apply(lambda x: x.sum(),axis=1)/133
    Y = [Rpt[i] - Rft[i] for i in range(len(Rpt))]
    X = [Rmt[i] - Rft[i] for i in range(len(Rmt))]
    X_addcons = sm.add_constant(X)
    model_port_sh_sz = sm.OLS(endog=Y, exog=X_addcons).fit()
    return model_port_sh_sz.summary()

def port_sh_func(data):
    '''
    对上交所的所有房地产股票跑回归
    :param data: 使用的数据集名称，当前为“data_new”
    :return:OLS回归结果
    '''
    Rit_dic = {}
    Rpt = []
    Rmt = np.log(data.iloc[:,1]/data.iloc[:,1].shift(1))
    Rmt = list(Rmt.dropna())
    Rft = list(data.iloc[1:,0]/52)
    for i in range(63,136):
        Rit = data.iloc[:, i]/data.iloc[:, i].shift(1)
        Rit = list(Rit.dropna())
        Rit_dic[i] = Rit
    Rit_table = pd.DataFrame(Rit_dic)
    Rpt = Rit_table.apply(lambda x: x.sum(),axis=1)/73
    Y = [Rpt[i] - Rft[i] for i in range(len(Rpt))]
    X = [Rmt[i] - Rft[i] for i in range(len(Rmt))]
    X_addcons = sm.add_constant(X)
    model_port_sh = sm.OLS(endog=Y, exog=X_addcons).fit()
    return model_port_sh.summary()

def port_sz_func(data):
    '''
    对深交所的所有房地产股票跑回归
    :param data: 使用的数据集名称，当前为“data_new”
    :return:OLS回归结果
    '''
    Rit_dic = {}
    Rpt = []
    Rmt = np.log(data.iloc[:,2]/data.iloc[:,2].shift(1))
    Rmt = list(Rmt.dropna())
    Rft = list(data.iloc[1:,0]/52)
    for i in range(3,63):
        Rit = data.iloc[:, i]/data.iloc[:, i].shift(1)
        Rit = list(Rit.dropna())
        Rit_dic[i] = Rit
    Rit_table = pd.DataFrame(Rit_dic)
    Rpt = Rit_table.apply(lambda x: x.sum(),axis=1)/133
    Y = [Rpt[i] - Rft[i] for i in range(len(Rpt))]
    X = [Rmt[i] - Rft[i] for i in range(len(Rmt))]
    X_addcons = sm.add_constant(X)
    model_port_sz = sm.OLS(endog=Y, exog=X_addcons).fit()
    return model_port_sz.summary()

def output_port():
    w = open(f"CAPM_result_by_port{datetime.datetime.now().strftime('%Y-%m-%d %H@%M@%S')}.csv", 'w+')
    w.write('port_sh_sz')
    w.write('\n\n')
    w.write(port_sh_sz_func(data_new).as_csv())
    w.write('\n\n\n\n')
    w.write('port_sh')
    w.write('\n\n')
    w.write(port_sh_func(data_new).as_csv())
    w.write('\n\n\n\n')
    w.write('port_sz')
    w.write('\n\n')
    w.write(port_sz_func(data_new).as_csv())
    w.write('\n\n\n\n')
    w.close()

output_port()
