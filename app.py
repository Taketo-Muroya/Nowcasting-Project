import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

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
#gt1 = gt1.rename(columns = {"失業": "unemployment", "isPartial": "info"})
#gt1.to_csv("gt1.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#gt1 = pd.read_csv('gt1.csv', index_col=0, date_parser=dateparse, dtype='float')

# Extract trend factor
s1 = seasonal_decompose(gt1.iloc[:,0], extrapolate_trend='freq')
t1 = s1.trend
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
#gt2 = gt2.rename(columns = {"貯金": "saving", "isPartial": "info"})
#gt2.to_csv("data/gt2.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#gt2 = pd.read_csv('data/gt2.csv', index_col=0, date_parser=dateparse, dtype='float')
st.table(gt2.tail(10))
st.line_chart(gt2.iloc[:,0])

# Extract trend factor
s2 = seasonal_decompose(gt2.iloc[:,0], extrapolate_trend='freq')
t2 = s2.trend
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
gtrend_l = pd.concat([t1, t2], axis=1)

# Combine google trend (YoY)
gtrend_y = pd.concat([a1, a2], axis=1).rename(columns={'unemployment': 'unemployment_rate', 'saving': 'saving_rate'})

# Set time series dataset
X = pd.merge(gtrend_l, gtrend_y, on='date')
y = ibc[228:]
y = y.set_index('time')
y.index = X.index
ts = pd.merge(y, X, on='date')

st.dataframe(ts)

#ts['date'] = pd.to_datetime(ts['date'])
#ts.set_index('date', inplace=True)

#st.dataframe(y.tail())
#st.dataframe(gtrend_l.tail())

#ts.to_csv("data/ts.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%b-%y')
#ts = pd.read_csv('data/ts.csv', index_col=0, date_parser=dateparse, dtype='float')

# set the dataset
#features = pd.concat([ts['Coincident Index'], ts.iloc[:,2:4]], axis=1)
#st.table(features.tail())

