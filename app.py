
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
# 景気動向指数のデータをHPからWebスクレイピングで自動取得する関数
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

# グーグルの検索情報をAPIを用いて自動取得する関数（月次）
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

  return data

# グーグルの検索情報をAPIを用いて自動取得する関数（週次）
def weekly_google_trend(kw):
  # Get the weekly google trend data
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

# 機械学習に備えたデータ整理をする関数
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

# 推計の精度を計算する関数
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

# 機械学習（LSTM-RNN）を実行する関数
def lstm_rnn(features):
  # set training percentage
  TRAIN_SPLIT = round(0.75*len(features))

  # feature scaling
  dataset = features.values
  data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
  data_std = dataset[:TRAIN_SPLIT].std(axis=0)
  dataset = (dataset-data_mean)/data_std

  # create the training and test data
  past_history = 2
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
  
  # Set the threshold
  if start == datetime.date(2004, 1, 1) and end == datetime.date.today() and kw1 == '失業' and kw2 == '貯金':
    test_score = 0
    while test_score < 0.8:
      # train the model
      single_step_history = single_step_model.fit(
        train_data_single, epochs=10, steps_per_epoch=200, validation_data=val_data_single, validation_steps=50
        )

      # evaluate the model
      model_eval_metrics(y_val_single, single_step_model.predict(x_val_single), classification="FALSE")
      test_score = r2_score(y_val_single, single_step_model.predict(x_val_single))

  else:
    # train the model
    single_step_history = single_step_model.fit(
      train_data_single, epochs=20, steps_per_epoch=200, validation_data=val_data_single, validation_steps=50
      )

    # evaluate the model
    model_eval_metrics(y_val_single, single_step_model.predict(x_val_single), classification="FALSE")
    test_score = r2_score(y_val_single, single_step_model.predict(x_val_single))
  
  # save the result
  predict = pd.DataFrame(single_step_model.predict(x_val_single)*data_std[0]+data_mean[0])
  predict.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

  actual = pd.DataFrame(y_val_single*data_std[0]+data_mean[0])
  actual.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

  output = pd.merge(predict, actual, on='date')
  output.columns = ['予測値', '実績値']

  return output, test_score, single_step_model

# 週次データによってナウキャスティングを実行する関数
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
    if end == datetime.date.today():
      st.write(XX.tail(8))
      st.write('-----------------------------------------------')

  # save the output
  future_estimate = pd.DataFrame(XX.iloc[END:len(XX)+1,0])
  df_concat = pd.concat([past_estimate.set_axis(['Coincident Index'], axis='columns'), future_estimate])

  return past_estimate, future_estimate, df_concat

# 本体 -------------------------------------------------------------------------------------
# サイドバー
st.sidebar.subheader("Google検索数による景気予測ツールです。")
kw1 = st.sidebar.text_input('検索ワードを記入してください', '失業')
kw2 = st.sidebar.text_input('検索ワードを記入してください', '貯金')
start = st.sidebar.date_input("データ開始時期を設定してください", datetime.datetime(2004, 1, 1))
end = st.sidebar.date_input("データ終了時期を設定してください", datetime.datetime.today())

# 景気動向指数とグーグル検索数を取得して統合
ibc = get_ibc_data('https://www.esri.cao.go.jp/jp/stat/di/')
data1 = google_trend(kw1)
data2 = google_trend(kw2)
X = pd.merge(data1, data2, on='date')
y = ibc[228:]
y = y.set_index('time')
y.index = X[:len(ibc)-228].index
ts = pd.merge(y, X, on='date')
latest_date, time = str(ts.index[-1]).split()

# データ期間の設定
ts = ts[(ts.index >= pd.to_datetime(start)) & (ts.index <= pd.to_datetime(end))]

# 本ページ
st.title('景気ナウキャスティング')
st.write("#####  ")
st.write("##### まず、景気動向指数とGoogle検索数の相関関係を確認します。検索ワードやデータ期間は左の記入欄から変更することができます。")
st.write("#####  ")

# グーグル検索数のグラフ１
st.write(f"""##### ● 景気動向指数と「{kw1}」のGoogle検索数""")
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(ts.index, ts.iloc[:,0], linestyle='-', color='b', label='Indexes of Business Conditions')
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(ts.index, ts.iloc[:,2], linestyle='--', color='#e46409', label='Google Search')
ax.plot(ts.index, ts.iloc[:,3], linestyle='-', color='b', label='Trend Element')
ax.legend()
st.pyplot(fig)

# 相関係数の計算１
cor_level1 = ts.iloc[:,0].corr(ts.iloc[:,3])
cor_ann1 = ts.iloc[:,1].corr(ts.iloc[:,4])
st.write("水準の相関係数：{:.2f}".format(cor_level1))
st.write("前年比の相関係数：{:.2f}".format(cor_ann1))

st.write("#####  ")

# グーグル検索数のグラフ２
st.write(f"""##### ● 景気動向指数と「{kw2}」のGoogle検索数""")
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(ts.index, ts.iloc[:,0], linestyle='-', color='b', label='Indexes of Business Conditions')
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(ts.index, ts.iloc[:,5], linestyle='--', color='#e46409', label='Google Search')
ax.plot(ts.index, ts.iloc[:,6], linestyle='-', color='b', label='Trend Element')
ax.legend()
st.pyplot(fig)

# 相関係数の計算２
cor_level2 = ts.iloc[:,0].corr(ts.iloc[:,6])
cor_ann2 = ts.iloc[:,1].corr(ts.iloc[:,7])
st.write("水準の相関係数：{:.2f}".format(cor_level2))
st.write("前年比の相関係数：{:.2f}".format(cor_ann2))

# 補足説明
st.caption(f'(※)「Indexes of Business Conditions」は景気動向指数の一致指数（最新月は{latest_date}が公表されている）。\
  「Google Search」はGoogle検索数を月次集計し指数化したもの。「Trend Element」はそのGoogle検索数のトレンド成分。\
    「水準の相関係数」は景気動向指数とトレンド成分の水準について相関係数を計算したもの。\
      「前年比の相関係数」は景気動向指数とトレンド成分の前年比について相関係数を計算したもの。')

# 推計開始ボタン
st.write('-----------------------------------------------')
st.write("##### 推計開始ボタンを押すと、Google検索数を用いて景気動向指数を推計します。")
st.write("#####  ")

# 推計 -------------------------------------------------------------------------------------
if st.button('推計開始'):
  comment = st.empty()
  comment.write('・・・推計中・・・')

  # 月次データによる推計
  ts = ts.drop(ts.columns[[1, 2, 4, 5, 7]], axis=1)
  output, test_score, single_step_model = lstm_rnn(ts)

  # 推計結果の図示
  st.write("""##### ● 推計された景気動向指数""")
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(output.index, output.iloc[:,1], linestyle='-', color='b', label='Actual')
  ax.plot(output.index, output.iloc[:,0], linestyle='--', color='#e46409', label='Predict')
  ax.legend()
  plt.xticks(rotation=30) 
  st.pyplot(fig)
  st.write(output.tail().T)
  st.write("Test set score（決定係数）: {:.2f}".format(test_score))
  
  # 補足説明
  st.caption(f'(※) {kw1}と{kw2}のGoogle検索数のトレンド成分と一期前の景気動向指数を用いて、当期の景気動向指数を推計している。\
    モデルはRNN-LSTM（Recurrent Neural Network - Long Short Term Memory）を使用している。')
  
  st.write('-----------------------------------------------')
  st.write("##### 次に、週次のGoogle検索数で景気動向指数をナウキャスティングします。")
  st.write("#####  ")

  # 週次のグーグル検索数の取得
  df1 = weekly_google_trend(kw1)
  df2 = weekly_google_trend(kw2)
  
  # 週次のグーグル検索数のグラフ
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
  if end == datetime.date.today():
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

  # Nowcastingの図示
  past_estimate, future_estimate, df_concat = nowcasting(XX)
  st.write(f"""##### ● 推計された景気動向指数（週次）""")
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(past_estimate.index, past_estimate, linestyle='-', color='b', label='Weekly IBC')
  ax.plot(future_estimate.index, future_estimate, linestyle='-', color='#e46409', label='Nowcasting')
  ax.legend()
  st.pyplot(fig)

  st.write('##### オレンジ色で表示されている部分が、最新のGoogle検索数によってナウキャスティングされた景気動向指数の予測値です。')

  # 補足説明
  latest_week, time = str(df_concat.index[-1]).split()
  st.caption(f'(※) 月次で推計した際のGoogle検索数（{kw1}及び{kw2}）のトレンド成分と景気動向指数の関係性に対して、\
    週次のGoogle検索数のトレンド成分と三期前までの週次の景気動向指数を当てはめ、景気動向指数（週次）を予測している。\
     モデルは同様にRNN-LSTM（Recurrent Neural Network - Long Short Term Memory）を使用している。{latest_week}までの予測が可能となっている。')

  comment.write('推計完了') 
