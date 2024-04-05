# -*- coding: utf-8 -*-
"""
金融資料視覺化看板

@author: 羅元廷
"""

# 載入必要模組
import os
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 


####### (1) 開始設定 #######
###### 設定網頁標題介面 
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化看板 </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)


###### 讀取資料
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def load_data(path):
    df = pd.read_pickle(path)
    return df
# ##### 讀取 excel 檔
# df_original = pd.read_excel("kbars_2330_2022-01-01-2022-11-18.xlsx")

##### 读取Pickle文件
df_original = load_data('kbars_2330_2022-01-01-2022-11-18.pkl')
# df_original = pd.read_pickle('kbars_2330_2022-01-01-2022-11-18.pkl')
#df.columns  ## Index(['Unnamed: 0', 'time', 'open', 'low', 'high', 'close', 'volume','amount'], dtype='object')
df_original = df_original.drop('Unnamed: 0',axis=1)



###### 選擇資料區間
st.subheader("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
# 使用条件筛选选择时间区间的数据
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]


####### (2) 轉化為字典 #######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def To_Dictionary_1(df):
    KBar_dic = df.to_dict()
    KBar_open_list = list(KBar_dic['open'].values())
    KBar_dic['open']=np.array(KBar_open_list)
    
    KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
    #KBar_dic['product'].size   ## 1596
    #KBar_dic['product'][0]      ## 'tsmc'
    
    KBar_time_list = list(KBar_dic['time'].values())
    KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
    KBar_dic['time']=np.array(KBar_time_list)
    
    KBar_low_list = list(KBar_dic['low'].values())
    KBar_dic['low']=np.array(KBar_low_list)
    
    KBar_high_list = list(KBar_dic['high'].values())
    KBar_dic['high']=np.array(KBar_high_list)
    
    KBar_close_list = list(KBar_dic['close'].values())
    KBar_dic['close']=np.array(KBar_close_list)
    
    KBar_volume_list = list(KBar_dic['volume'].values())
    KBar_dic['volume']=np.array(KBar_volume_list)
    
    KBar_amount_list = list(KBar_dic['amount'].values())
    KBar_dic['amount']=np.array(KBar_amount_list)
    
    return KBar_dic

KBar_dic = To_Dictionary_1(df)


#######  (3) 改變 KBar 時間長度 & 形成 KBar 字典 (新週期的)  #######
###### 定義函數: 進行 K 棒更新  &  形成 KBar 字典 (新週期的): 設定cycle_duration可以改成你想要的 KBar 週期
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Change_Cycle(Date,cycle_duration,KBar_dic):
    ###### 進行 K 棒更新
    KBar = indicator_forKBar_short.KBar(Date,cycle_duration)    ## 設定cycle_duration可以改成你想要的 KBar 週期
    for i in range(KBar_dic['time'].size):
        #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
        time = KBar_dic['time'][i]
        #prod = KBar_dic['product'][i]
        open_price= KBar_dic['open'][i]
        close_price= KBar_dic['close'][i]
        low_price= KBar_dic['low'][i]
        high_price= KBar_dic['high'][i]
        qty =  KBar_dic['volume'][i]
        amount = KBar_dic['amount'][i]
        #tag=KBar.TimeAdd(time,price,qty,prod)
        tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    ###### 形成 KBar 字典 (新週期的):
    KBar_dic = {}
    KBar_dic['time'] =  KBar.TAKBar['time']   
    #KBar_dic['product'] =  KBar.TAKBar['product']
    KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
    KBar_dic['open'] = KBar.TAKBar['open']
    KBar_dic['high'] =  KBar.TAKBar['high']
    KBar_dic['low'] =  KBar.TAKBar['low']
    KBar_dic['close'] =  KBar.TAKBar['close']
    KBar_dic['volume'] =  KBar.TAKBar['volume']
    
    return KBar_dic
    

###### 改變日期資料型態
Date = start_date.strftime("%Y-%m-%d")

###### 設定 K 棒的時間長度(分鐘)
st.subheader("設定一根 K 棒的時間長度(分鐘)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', key="KBar_duration")
cycle_duration = int(cycle_duration)

###### 進行 K 棒更新  & 形成 KBar 字典 (新週期的)
KBar_dic = Change_Cycle(Date,cycle_duration,KBar_dic)   ## 設定cycle_duration可以改成你想要的 KBar 週期



####### (4) 計算各種技術指標 #######
###### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)

######  (i) 移動平均線策略 
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MA(df, period=10):
    ##### 計算長短移動平均線
    ma = df['close'].rolling(window=period).mean()
    return ma
  
#####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)")
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)")
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

##### 計算長短移動平均線
KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)

##### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]



######  (ii) RSI 策略 
##### 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
  
##### 順勢策略
#### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K棒週期數目(整數, 例如 10)")
LongRSIPeriod=st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K棒週期數目(整數, 例如 2)")
ShortRSIPeriod=st.slider('選擇一個整數', 0, 1000, 2)

#### 計算 RSI指標長短線, 以及定義中線
KBar_df['RSI_long'] = Calculate_RSI(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = Calculate_RSI(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

#### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]


# ##### 逆勢策略
# #### 建立部位管理物件
# OrderRecord=Record() 
# #### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# #### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)


######  (iii) Bollinger Band (布林通道) 策略 
##### 假设df是包含价格数据的Pandas DataFrame，'close'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df


#####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)")
period = st.slider('選擇一個整數', 0, 100, 20, key='BB_period')
st.subheader("設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)")
num_std_dev = st.slider('選擇一個整數', 0, 100, 2, key='BB_heigh')

##### 計算布林通道上中下通道:
KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)

##### 尋找最後 NAN值的位置
last_nan_index_BB = KBar_df['SMA'][::-1].index[KBar_df['SMA'][::-1].apply(pd.isna)][0]


######  (iv) MACD(異同移動平均線) 策略 
# 假设df是包含价格数据的Pandas DataFrame，'price'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']  ## DIF
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()   ## DEA或信號線
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']  ## MACD = DIF-DEA
    return df

#####  設定MACD三種週期的K棒長度:
st.subheader("設定計算 MACD的快速線週期(例如 12根日K)")
fast_period = st.slider('選擇一個整數', 0, 100, 12)
st.subheader("設定計算 MACD的慢速線週期(例如 26根日K)")
slow_period = st.slider('選擇一個整數', 0, 100, 26)
st.subheader("設定計算 MACD的訊號線週期(例如 9根日K)")
signal_period = st.slider('選擇一個整數', 0, 100, 9)

##### 計算MACD:
KBar_df = Calculate_MACD(KBar_df, fast_period, slow_period, signal_period)

##### 尋找最後 NAN值的位置
# last_nan_index_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)][0]
#### 試著找出最後一個 NaN 值的索引，但在這之前要檢查是否有 NaN 值
nan_indexes_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)]
if len(nan_indexes_MACD) > 0:
    last_nan_index_MACD = nan_indexes_MACD[0]
else:
    last_nan_index_MACD = 0




####### (5) 將 Dataframe 欄位名稱轉換(第一個字母大寫)  ####### 
KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


####### (6) 畫圖 #######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
import plotly.offline as pyoff


###### K線圖, 移動平均線MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


###### K線圖, RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)
    

###### K線圖, Bollinger Band    
with st.expander("K線圖, 布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)    
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['SMA'][last_nan_index_BB+1:], mode='lines',line=dict(color='black', width=2), name='布林通道中軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['Upper_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='red', width=2), name='布林通道上軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['Lower_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='blue', width=2), name='布林通道下軌道'), 
                  secondary_y=False)
    
    fig3.layout.yaxis2.showgrid=True

    st.plotly_chart(fig3, use_container_width=True)



###### MACD
with st.expander("MACD(異同移動平均線)"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # #### include candlestick with rangeselector
    # fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
    #                 open=KBar_df['Open'], high=KBar_df['High'],
    #                 low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
    #                secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig4.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['Signal_Line'][last_nan_index_MACD+1:], mode='lines',line=dict(color='orange', width=2), name='訊號線(DEA)'), 
                  secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['MACD'][last_nan_index_MACD+1:], mode='lines',line=dict(color='pink', width=2), name='DIF'), 
                  secondary_y=True)
    
    fig4.layout.yaxis2.showgrid=True
    st.plotly_chart(fig4, use_container_width=True)















