import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import tensorflow as tf
import statsmodels.api as sm

#from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

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


# Import packages
from pytrends.request import TrendReq
plt.rcParams['font.family'] = 'IPAexGothic'

# API Connection
pytrends = TrendReq(hl='ja-JP', tz=360)

st.title('景気ナウキャスティング')

ibc = pd.read_csv('data/ibc_new.csv')
ibc['Coincident ann'] = 100*ibc['Coincident Index'].pct_change(12)
#st.table(ibc.tail(10))
#st.line_chart(ibc['Coincident Index'])

#st.sidebar.write("""
## GAFA株価
#こちらは株価可視化ツールです。以下のオプションから表示日数を指定できます。
#""")

kw1 = st.sidebar.text_input('検索ワードを記入してください', '失業')
kw2 = st.sidebar.text_input('検索ワードを記入してください', '貯金')

st.write(f"""
### **「{kw1}」** のグーグルトレンド
""")

# Set keyword ("失業" = "unemployment")
#@st.cache
kw_list1 = [kw1]
pytrends.build_payload(kw_list1, timeframe='2004-01-01 2021-11-30', geo='JP')
gt1 = pytrends.interest_over_time()

st.table(gt1.tail(10))
st.line_chart(gt1.iloc[:,0])
gt1 = gt1.rename(columns = {kw1:"var1", "isPartial":"info"})
#gt1.to_csv("gt1.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#gt1 = pd.read_csv('gt1.csv', index_col=0, date_parser=dateparse, dtype='float')

# Extract trend factor
t1 = sm.tsa.seasonal_decompose(gt1.iloc[:,0], extrapolate_trend='freq').trend
st.line_chart(t1)
#plt.plot(t1)
#plt.plot(gt1.iloc[:,0], linestyle='--')

# Check correlation
level = ibc['Coincident Index'][228:]
level.index = t1.index
cor = level.corr(t1)
st.write("Correlation of level: {:.2f}".format(cor))
#print("Correlation of level: {:.2f}".format(cor))

a1 = gt1.iloc[:,0].pct_change(12)
ann = ibc['Coincident ann'][228:]
ann.index = a1.index
cor = ann.corr(a1)
st.write("Correlation of YoY: {:.2f}".format(cor))
#print("Correlation of YoY: {:.2f}".format(cor))

st.write(f"""
### **「{kw2}」** のグーグルトレンド
""")

# Set keyword ("貯金" = "saving")
#@st.cache
kw_list2 = [kw2]
pytrends.build_payload(kw_list2, timeframe='2004-01-01 2021-11-30', geo='JP')
gt2 = pytrends.interest_over_time()
gt2 = gt2.rename(columns = {kw2:"var2", "isPartial":"info"})
#gt2.to_csv("data/gt2.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#gt2 = pd.read_csv('data/gt2.csv', index_col=0, date_parser=dateparse, dtype='float')
st.table(gt2.tail(10))
st.line_chart(gt2.iloc[:,0])

# Extract trend factor
t2 = seasonal_decompose(gt2.iloc[:,0], extrapolate_trend='freq').trend
st.line_chart(t2)
#plt.plot(t2)
#plt.plot(gt2.iloc[:,0], linestyle='--')

# Check correlation
level = ibc['Coincident Index'][228:]
level.index = t2.index
cor = level.corr(t2)
st.write("Correlation of level: {:.2f}".format(cor))

a2 = gt2.iloc[:,0].pct_change(12)
ann = ibc['Coincident ann'][228:]
ann.index = a2.index
cor = ann.corr(a2)
st.write("Correlation of YoY: {:.2f}".format(cor))

# Combine google trend (level)
t1 = pd.DataFrame(t1).rename(columns = {"trend":"trend-1"})
t2 = pd.DataFrame(t2).rename(columns = {"trend":"trend-2"})
gtrend_l = pd.concat([t1, t2], axis=1)

# Combine google trend (YoY)
gtrend_y = pd.concat([a1, a2], axis=1).rename(columns={'var1': 'var1_rate', 'var2': 'var2_rate'})

# Set time series dataset
X = pd.merge(gtrend_l, gtrend_y, on='date')
y = ibc[228:]
y = y.set_index('time')
y.index = X.index
ts = pd.merge(y, X, on='date')

st.dataframe(ts)

if st.button('Estimation'):

  # set the dataset
  features = pd.concat([ts['Coincident Index'], ts.iloc[:,2:4]], axis=1)
  #features.tail()

  # set training percentage
  TRAIN_SPLIT = round(0.8*len(features))
  #print(TRAIN_SPLIT)

  # feature scaling
  dataset = features.values
  data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
  data_std = dataset[:TRAIN_SPLIT].std(axis=0)
  dataset = (dataset-data_mean)/data_std

  # create the training and test data
  past_history = 3
  future_target = 0
  STEP = 1

  x_train_single, y_train_single = multivariate_data(dataset, dataset[:,0], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
  x_val_single, y_val_single = multivariate_data(dataset, dataset[:,0], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)

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

  # train the model
  single_step_history = single_step_model.fit(train_data_single, epochs=10, steps_per_epoch=200, validation_data=val_data_single, validation_steps=50)

  # evaluate the model
  model_eval_metrics(y_val_single, single_step_model.predict(x_val_single), classification="FALSE")

  # visualize the result
  predict = pd.DataFrame(single_step_model.predict(x_val_single)*data_std[0]+data_mean[0])
  predict.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

  actual = pd.DataFrame(y_val_single*data_std[0]+data_mean[0])
  actual.index = features.iloc[TRAIN_SPLIT+past_history:,:].index

  st.line_chart(predict)
  st.line_chart(actual)

  #plt.plot(features.iloc[TRAIN_SPLIT+past_history:,0], label="actual")
  #plt.plot(predict, "r", linestyle='--', label="predict")
  #plt.legend(loc='best')
  #plt.title('RNN-LSTM: Level')
  #plt.savefig("images/google_lstm.png")

  st.write("Test set score: {:.2f}".format(r2_score(y_val_single, single_step_model.predict(x_val_single))))


  # Get the weekly google trend data (unemployment)
  pytrends.build_payload(kw_list1, timeframe='today 5-y', geo='JP')
  #pytrends.build_payload(kw_list, timeframe='2017-01-01 2021-01-16', geo='JP')
  gt3 = pytrends.interest_over_time()
  gt3 = gt3.rename(columns = {kw1:"var1", "isPartial":"info"})
  #gt3.to_csv("data/gt3.csv")
  #dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
  #gt3 = pd.read_csv('data/gt3.csv', index_col=0, date_parser=dateparse, dtype='float')

  # Extract trend factor
  s3 = seasonal_decompose(gt3.iloc[:,0], freq=6, extrapolate_trend='freq')
  t3 = s3.trend
  #gtw_u = pd.DataFrame(t3)
  #gtw_u.to_csv("data/gtw_u.csv")

  st.line_chart(t3)
  st.line_chart(gt3.iloc[:,0])


  # Get the weekly google trend data (saving)
  pytrends.build_payload(kw_list2, timeframe='today 5-y', geo='JP')
  #pytrends.build_payload(kw_list, timeframe='2004-01-01 2020-02-29', geo='JP')
  gt4 = pytrends.interest_over_time()
  gt4 = gt4.rename(columns = {kw2: "var2", "isPartial": "info"})
  #gt4.to_csv("data/gt4.csv")
  #dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
  #gt4 = pd.read_csv('data/gt4.csv', index_col=0, date_parser=dateparse, dtype='float')

  # Extract trend factor
  s4 = seasonal_decompose(gt4.iloc[:,0], freq=24, extrapolate_trend='freq')
  t4 = s4.trend
  #gtw_s = pd.DataFrame(t4)
  #gtw_s.to_csv("data/gtw_s.csv")

  st.line_chart(t4)
  st.line_chart(gt4.iloc[:,0])

  # load the weekly ibc data
  dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
  wibc = pd.read_csv('data/wibc.csv', index_col=0, date_parser=dateparse, dtype='float')

  # merge google trend with ibc data
  temp = pd.merge(t3, t4, on='date')
  XX = pd.merge(wibc, temp, on='date')
  st.table(XX.tail(10))

  # feature scaling
  END = len(XX)-XX['ibc'].isnull().sum()
  dataset = XX.iloc[:END,:].values
  data_mean = dataset.mean(axis=0)
  data_std = dataset.std(axis=0)
  dataset = (dataset-data_mean)/data_std

  # create the test data
  x_single, y_single = multivariate_data(dataset, dataset[:,0], 0, None, past_history, future_target, STEP, single_step=True)

  # save the output
  past_estimate = pd.DataFrame(single_step_model.predict(x_single)*data_std[0]+data_mean[0])
  past_estimate.index = XX.iloc[past_history:END,:].index

  # visualize the result 
  st.line_chart(past_estimate)

  # nowcast the future IBC
  for i in range(END, len(XX)):
    XX.iat[i,0] = float(single_step_model.predict(x_single)[-1]*data_std[0]+data_mean[0])
    #XX.iat[i,0] = XX.iat[i-1,0]
    temp = XX.iloc[:i+1,:]
    st.write(temp.tail())
    st.write('-----------------------------------------------')

    # feature scaling
    dataset = temp.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    dataset = (dataset-data_mean)/data_std
    
    # create the test data
    x_single, y_single = multivariate_data(dataset, dataset[:,0], 0, None, past_history, future_target, STEP, single_step=True)

    XX.iat[i,0] = float(single_step_model.predict(x_single)[-1]*data_std[0]+data_mean[0])
    st.write(XX.tail(10))
    st.write('-----------------------------------------------')

  # save the output
  future_estimate = pd.DataFrame(XX.iloc[END:len(XX)+1,0])

  #plt.plot(single_step_model.predict(x_single)*data_std[0]+data_mean[0], "r", linestyle='--', label="predict")
  df_concat = pd.concat([past_estimate.set_axis(['ibc'], axis='columns'), future_estimate])
  st.line_chart(df_concat)

