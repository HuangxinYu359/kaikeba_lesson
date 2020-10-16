from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
import calendar
from datetime import datetime,timedelta

warnings.filterwarnings('ignore')
data = pd.read_csv('G:/project_courseware/核心班BI/lesson08/002621_1990_12_19_to_2020_10_15.csv',encoding = 'GB2312')
data = data[['日期','收盘价']]
#将日期作为索引
data['日期'] = pd.to_datetime(data['日期'])
data.index = data['日期']

#将几天市场价为0的日子进行插值
data['收盘价'].replace(0,None,inplace = True)
data['收盘价'].interpolate(inplace=True)

#按照月来统计
data_month = data.resample('M').mean()
data_Q = data.resample('Q-DEC').mean()
data_Y = data.resample('A-DEC').mean()

#按天、月、季度、年显示股票走势
fig = plt.figure(figsize=[20,5])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.suptitle('股票走势图',fontsize = 20)
plt.subplot(221)
plt.plot(data['收盘价'],'-',label = '按天')
plt.legend()
plt.subplot(222)
plt.plot(data_month['收盘价'],'-',label = '按月')
plt.legend()
plt.subplot(223)
plt.plot(data_Q['收盘价'],'-',label = '按季度')
plt.legend()
plt.subplot(224)
plt.plot(data_Y['收盘价'],'-',label = '按年')
plt.legend()
plt.show()

#设置参数范围
ps = range(0,3)
qs = range(0,3)
parameters = product(ps,qs)
parameters_list = list(parameters)

#寻找最优ARMA模型参数，AIC最小
results = []
best_aic = float('inf')
for param in parameters_list:
    try:
        model =ARMA(data_month['收盘价'],order=(param[0],param[1])).fit()
    except ValueError:
        print('参数错误：',param)
        continue
    aic = model.aic
    if aic<best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param,model.aic])

#输出最优模型
print('最优模型:',best_model.summary())

#设置future_month,需要预测的时间date_list
data_month2 = data_month[['收盘价']]
future_month = 3
last_month = pd.to_datetime(data_month2.index[len(data_month2)-1])
#print(last_month)
date_list = []
for i in range(future_month):
    #计算下个月有几天
    year = last_month.year
    month = last_month.month
    if month == 12:
        month = 1
        year +=1
    else:
        month +=1
    next_month_days = calendar.monthrange(year,month)[1]
    last_month =last_month +timedelta(days = next_month_days)
    date_list.append(last_month)
print('date_list',date_list)

#添加未来要预测的3个月
future = pd.DataFrame(index = date_list,columns=data_month.columns)
data_month2 = pd.concat([data_month2,future])
data_month2['forecast'] = best_model.predict(start=0,end=len(data_month2))
#第一个元素不正确，设置为NaN
data_month2['forecast'][0] = np.NaN
print(data_month2)

#股票显示
plt.figure(figsize=(30,7))
data_month2['收盘价'].plot(label='实际指数')
data_month2.forecast.plot(color='r', ls='--', label='预测指数')
plt.legend()
plt.title('沪市指数（月）')
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()





