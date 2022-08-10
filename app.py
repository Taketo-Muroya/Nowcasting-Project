
# 設定 -------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import tensorflow as tf
import statsmodels.api as sm
import requests
import datetime

from math import sqrt
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'

# API Connection
pytrends = TrendReq(hl='ja-JP', tz=360)

# 関数 -------------------------------------------------------------------------------------
def get_ibc_data(url):
  url_index = url + 'di.html'
  res = requests.get(url_index)
  soup = BeautifulSoup(res.text, 'html.parser')
  name = soup.find_all('a', {'target': '_blank'})[1].attrs['href']

  if 'xlsx' in name: # 確報公表時
    input_file_name = url + name
    input_book = pd.ExcelFile(input_file_name)
    input_sheet_name = input_book.sheet_names
    input_sheet_df = input_book.parse(input_sheet_name[0], skiprows=3)
    input_sheet_df = input_sheet_df.iloc[62:,[0,4]]
    input_sheet_df = input_sheet_df.rename(columns={'Time (Monthly) Code':'time'})
    input_sheet_df['time'] = input_sheet_df['time'].astype('int')
    ibc = input_sheet_df.astype('float')
    ibc['Coincident ann'] = 100*ibc['Coincident Index'].pct_change(12)

  else: # 速報公表時
    name = soup.find_all('a', {'target': '_blank'})[2].attrs['href']
    input_file_name = url + name
    input_book = pd.ExcelFile(input_file_name)
    input_sheet_name = input_book.sheet_names
    input_sheet_df = input_book.parse(input_sheet_name[0], skiprows=3)
    input_sheet_df = input_sheet_df.iloc[62:,[0,4]]
    input_sheet_df = input_sheet_df.rename(columns={'Time (Monthly) Code':'time'})
    input_sheet_df['time'] = input_sheet_df['time'].astype('int')
    ibc = input_sheet_df.astype('float')
    ibc['Coincident ann'] = 100*ibc['Coincident Index'].pct_change(12)
  
  return ibc

def google_trend(kw):
  #@st.cache
  kw_list = [kw]
  pytrends.build_payload(kw_list, timeframe='all', geo='JP')
  gt = pytrends.interest_over_time()
  gt = gt.rename(columns = {kw:"variable", "isPartial":"info"})

  # Extract trend factor and YoY
  t = seasonal_decompose(gt.iloc[:,0], extrapolate_trend='freq', period=12).trend
  #t = pd.DataFrame(t).rename(columns = {"trend":f"{kw}-trend"})
  a = gt.iloc[:,0].pct_change(12)
  #a = pd.DataFrame(a).rename(columns = {"variable":f"{kw}-YoY"})
  temp = pd.merge(gt.iloc[:,0], t, on='date')
  data = pd.merge(temp, a, on='date')

  # Check correlation
  level = ibc['Coincident Index'][228:]
  level.index = t[:len(ibc)-228].index
  cor_level = level.corr(t[:len(ibc)-228])
  ann = ibc['Coincident ann'][228:]
  ann.index = a[:len(ibc)-228].index
  cor_ann = ann.corr(a[:len(ibc)-228])

  return data, cor_level, cor_ann

def weekly_google_trend(kw):
  # Get the weekly google trend data (unemployment)
  kw_list = [kw]
  pytrends.build_payload(kw_list, timeframe='today 5-y', geo='JP')
  #pytrends.build_payload(kw_list, timeframe='2017-01-01 2021-01-16', geo='JP')
  gt = pytrends.interest_over_time()
  gt = gt.rename(columns = {kw:"variable", "isPartial":"info"})
 
  # Extract trend factor
  s = seasonal_decompose(gt.iloc[:,0], extrapolate_trend='freq', period=12)
  t = s.trend
  data = pd.merge(gt.iloc[:,0], t, on='date')

  return data

def multivariate_data(dataset, target, start_index, end_index, 
                      history_size, target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    # add current dataset
    indices = range(i-history_size, i+1, step)
    temp = pd.DataFrame(dataset[indices])
    # replace current target to previous one
    temp.iat[history_size, 0] = temp.iat[history_size-1, 0]
    data.append(np.array(temp))

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def model_eval_metrics(y_true, y_pred, classification="TRUE"):
     if classification=="TRUE":
        accuracy_eval = accuracy_score(y_true, y_pred)
        f1_score_eval = f1_score(y_true, y_pred,average="macro",zero_division=0)
        precision_eval = precision_score(y_true, y_pred,average="macro",zero_division=0)
        recall_eval = recall_score(y_true, y_pred,average="macro",zero_division=0)
        mse_eval = 0
        rmse_eval = 0
        mae_eval = 0
        r2_eval = 0
        metricdata = {'accuracy': [accuracy_eval], 'f1_score': [f1_score_eval], 
                      'precision': [precision_eval], 'recall': [recall_eval], 'mse': [mse_eval], 
                      'rmse': [rmse_eval], 'mae': [mae_eval], 'r2': [r2_eval]}
        finalmetricdata = pd.DataFrame.from_dict(metricdata)
     else:
        accuracy_eval = 0
        f1_score_eval = 0
        precision_eval = 0
        recall_eval = 0
        mse_eval = mean_squared_error(y_true, y_pred)
        rmse_eval = sqrt(mean_squared_error(y_true, y_pred))
        mae_eval = mean_absolute_error(y_true, y_pred)
        r2_eval = r2_score(y_true, y_pred)
        metricdata = {'accuracy': [accuracy_eval], 'f1_score': [f1_score_eval], 
                      'precision': [precision_eval], 'recall': [recall_eval], 'mse': [mse_eval], 
                      'rmse': [rmse_eval], 'mae': [mae_eval], 'r2': [r2_eval]}
        finalmetricdata = pd.DataFrame.from_dict(metricdata)
     return finalmetricdata

def lstm_rnn(features):
  # set training percentage
  TRAIN_SPLIT = round(0.7*len(features))

  # feature scaling
  dataset = features.values
  data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
  data_std = dataset[:TRAIN_SPLIT].std(axis=0)
  dataset = (dataset-data_mean)/data_std

  # create the training and test data
  past_history = 1
  future_target = 0
  STEP = 1

  x_train_single, y_train_single = multivariate_data(
    dataset, dataset[:,0], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True
    )
  x_val_single, y_val_single = multivariate_data(
    dataset, dataset[:,0], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True
    )

  BATCH_SIZE = 32
  BUFFER_SIZE = 100

  train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
  train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
  val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

  # construct the model
  single_step_model = tf.keras.models.Sequential()
  single_step_model.add(tf.keras.layers.LSTM(8, input_shape=x_train_single.shape[-2:]))
  #single_step_model.add(tf.keras.layers.LSTM(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
  #single_step_model.add(tf.keras.layers.LSTM(8, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
  #single_step_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)))
  single_step_model.add(tf.keras.layers.Dense(1))

  single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='mae')

  test_score = 0
  while test_score < 0.8:
    # train the model
    single_step_history = single_step_model.fit(
      train_data_single, epochs=10, steps_per_epoch=200, validation_data=val_data_single, validation_steps=50
      )

    # evaluate the model
    model_eval_metrics(y_val_single, single_step_model.predict(x_val_single), classification="FALSE")

    # visualize the result
    predict = pd.DataFrame(single_step_model.predict(x_val_single)*data_std[0]+data_mean[0])
    predict.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

    actual = pd.DataFrame(y_val_single*data_std[0]+data_mean[0])
    actual.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

    output = pd.merge(predict, actual, on='date')
    output.columns=['予測値', '実績値']
    test_score = r2_score(y_val_single, single_step_model.predict(x_val_single))

  return output, test_score, single_step_model

def nowcasting(XX):

  # feature scaling
  END = len(XX)-XX['Coincident Index'].isnull().sum()
  dataset = XX.iloc[:END,:].values
  data_mean = dataset.mean(axis=0)
  data_std = dataset.std(axis=0)
  dataset = (dataset-data_mean)/data_std

  # create the test data
  past_history = 3
  future_target = 0
  STEP = 1
  x_single, y_single = multivariate_data(
    dataset, dataset[:,0], 0, None, past_history, future_target, STEP, single_step=True)

  # save the output
  past_estimate = pd.DataFrame(single_step_model.predict(x_single)*data_std[0]+data_mean[0])
  past_estimate.index = XX.iloc[past_history:END,:].index

  # nowcast the future IBC
  for i in range(END, len(XX)):
    XX.iat[i,0] = float(single_step_model.predict(x_single)[-1]*data_std[0]+data_mean[0])
    #XX.iat[i,0] = XX.iat[i-1,0]
    temp = XX.iloc[:i+1,:]
    #st.write(temp.tail())
    #st.write('-----------------------------------------------')

    # feature scaling
    dataset = temp.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    dataset = (dataset-data_mean)/data_std
    
    # create the test data
    x_single, y_single = multivariate_data(
      dataset, dataset[:,0], 0, None, past_history, future_target, STEP, single_step=True)

    XX.iat[i,0] = float(single_step_model.predict(x_single)[-1]*data_std[0]+data_mean[0])
    XX.columns=['景気動向指数', f'{kw1}のトレンド', f'{kw2}のトレンド']
    st.write(XX.tail(8))
    st.write('-----------------------------------------------')

  # save the output
  future_estimate = pd.DataFrame(XX.iloc[END:len(XX)+1,0])
  df_concat = pd.concat([past_estimate.set_axis(['Coincident Index'], axis='columns'), future_estimate])

  return past_estimate, future_estimate, df_concat

# 本体 -------------------------------------------------------------------------------------

# サイドバー
st.sidebar.write("""Google検索数による景気予測ツールです。検索ワードを記入してください。""")
kw1 = st.sidebar.text_input('検索ワードを記入してください', '失業')
kw2 = st.sidebar.text_input('検索ワードを記入してください', '貯金')
start = st.sidebar.date_input("どの期間からのデータを使用しますか？", datetime.date(2004, 1, 1))
end = st.sidebar.date_input("どの期間までのデータを使用しますか？", datetime.date.today())

# 景気動向指数とグーグル検索数の統合
ibc = get_ibc_data('https://www.esri.cao.go.jp/jp/stat/di/')
data1, cor_level1, cor_ann1 = google_trend(kw1)
data2, cor_level2, cor_ann2 = google_trend(kw2)

X = pd.merge(data1.iloc[:,1], data2.iloc[:,1], on='date')
y = ibc[228:]
y = y.set_index('time')
y.index = X[:len(ibc)-228].index
ts = pd.merge(y, X, on='date')
ts = ts.drop('Coincident ann', axis=1)

st.title('景気ナウキャスティング')
st.write("#####  ")
st.write("##### まず、景気動向指数とGoogle検索数の相関関係を確認します。検索ワードは左の記入欄から変更することができます。")
st.write("#####  ")

# グーグル検索数のグラフ
st.write(f"""##### ● 景気動向指数と「{kw1}」のGoogle検索数""")
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(ts.index, ts['Coincident Index'], linestyle='-', color='b', label='Indexes of Business Conditions')
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(data1.index, data1.iloc[:,1], linestyle='-', color='b', label='Trend Element')
ax.plot(data1.index, data1.iloc[:,0], linestyle='--', color='#e46409', label='Google Search')
ax.legend()
st.pyplot(fig)
st.write("水準の相関関数：{:.2f}".format(cor_level1))
st.write("前年比の相関関数：{:.2f}".format(cor_ann1))

st.write('-----------------------------------------------')

st.write(f"""##### ● 景気動向指数と「{kw2}」のGoogle検索数""")
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(ts.index, ts['Coincident Index'], linestyle='-', color='b', label='Indexes of Business Conditions')
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(data2.index, data2.iloc[:,1], linestyle='-', color='b', label='Trend Element')
ax.plot(data2.index, data2.iloc[:,0], linestyle='--', color='#e46409', label='Google Search')
ax.legend()
st.pyplot(fig)
st.write("水準の相関関数：{:.2f}".format(cor_level2))
st.write("前年比の相関関数：{:.2f}".format(cor_ann2))


st.write('-----------------------------------------------')
st.write("##### 推計開始ボタンを押すと、Google検索数を用いて景気動向指数を推計します。")
st.write("#####  ")

from datetime import datetime
dt = datetime.now()
st.write(ts)
st.write(ts.index)
st.write(ts.index.date())

ts = ts[ts.index > datetime.datetime(2010, 1, 1)]
#ts.index = ts.index.datatime.date()
#ts.index = pd.to_datetime(ts.index).date()
#st.dataframe(datatime.date(ts.index))

# 推計 -------------------------------------------------------------------------------------
if st.button('推計開始'):
  comment = st.empty()
  comment.write('・・・推計中・・・')

  # 月次データによる推計
  output, test_score, single_step_model = lstm_rnn(ts)

  st.write(f"""##### ● 推計された景気動向指数""")
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(output.index, output.iloc[:,1], linestyle='-', color='b', label='Actual')
  ax.plot(output.index, output.iloc[:,0], linestyle='--', color='#e46409', label='Predict')
  ax.legend()
  plt.xticks(rotation=30) 
  st.pyplot(fig)
  st.write(output.tail().T)
  st.write("Test set score（決定係数）: {:.2f}".format(test_score))
  
  st.write('-----------------------------------------------')
  st.write("##### 次に、週次のGoogle検索数で景気動向指数をナウキャスティングします。")
  st.write("#####  ")

  # 週次のグーグル検索数の取得
  df1 = weekly_google_trend(kw1)
  df2 = weekly_google_trend(kw2)

  st.write(f"""##### ● 「{kw1}」&「{kw2}」のGoogle検索数（週次）""")
  fig = plt.figure()
  ax = fig.add_subplot(2, 1, 1)
  ax.plot(df1.index, df1.iloc[:,1], linestyle='-', color='b', label='Trend Element')
  ax.plot(df1.index, df1.iloc[:,0], linestyle='--', color='#e46409', label='Google Search')
  ax.legend()
  ax = fig.add_subplot(2, 1, 2)
  ax.plot(df2.index, df2.iloc[:,1], linestyle='-', color='b', label='Trend Element')
  ax.plot(df2.index, df2.iloc[:,0], linestyle='--', color='#e46409', label='Google Search')
  ax.legend()
  st.pyplot(fig)

  st.write("#####  ")
  st.write("##### 以下、推計プロセス。")
  st.write("#####  ")

  # 週次の景気動向指数とグーグル検索数の統合
  temp1 = ts
  temp1['monthly'] = ts.index.year.astype('str') + '-' + ts.index.month.astype('str')
  temp2 = pd.merge(df1.iloc[:,1], df2.iloc[:,1], on='date')
  temp2['monthly'] = temp2.index.year.astype('str') + '-' + temp2.index.month.astype('str')
  temp3 = temp1.reset_index().set_index('monthly')
  temp4 = temp2.reset_index().set_index('monthly')
  temp5 = pd.merge(temp3, temp4, on='monthly', how='right')
  XX = temp5[['date_y','Coincident Index','trend_x_y','trend_y_y']].set_index('date_y')

  # Nowcasting
  past_estimate, future_estimate, df_concat = nowcasting(XX)
  st.write(f"""##### ● 推計された景気動向指数（週次）""")
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(past_estimate.index, past_estimate, linestyle='-', color='b', label='Weekly IBC')
  ax.plot(future_estimate.index, future_estimate, linestyle='-', color='#e46409', label='Nowcasting')
  ax.legend()
  st.pyplot(fig)
  #st.write(df_concat.tail().T)

  st.write('##### オレンジ色で表示されている部分が、最新のGoogle検索数によってナウキャスティングされた景気動向指数の予測値です。')

  comment.write('推計完了') 
