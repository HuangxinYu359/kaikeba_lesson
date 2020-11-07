from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv')

#转换为pandas中的日期格式
data['Datetime'] = pd.to_datetime(data['Datetime'],format = '%d-%m-%Y %H:%M')

#将Datetime作为data的索引
data.index = data['Datetime']
data.drop(['ID','Datetime'],axis=1,inplace = True)

#安装天进行重采样
daily_data = data.resample('D').sum()
daily_data['ds'] = daily_data.index
daily_data['y'] = daily_data['Count']
daily_data.drop(['Count'],axis =1,inplace = True)
print(daily_data)

m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
m.fit(daily_data)
# 预测未来7个月，213天
future = m.make_future_dataframe(periods=213)
forecast = m.predict(future)


m.plot(forecast)

#查看各个compinents
m.plot_components(forecast)

