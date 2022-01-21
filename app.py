import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Import packages
from pytrends.request import TrendReq
plt.rcParams['font.family'] = 'IPAexGothic'

# API Connection
pytrends = TrendReq(hl='ja-JP', tz=360)

st.title('グーグルトレンドによる景気ナウキャスティング')

st.sidebar.write("""
# GAFA株価
こちらは株価可視化ツールです。以下のオプションから表示日数を指定できます。
""")

kw = st.sidebar.text_input('検索ワードを記入してください')

st.write(f"""
### 過去 **{kw}日間** のGAFA株価
""")

#@st.cache
ibc = pd.read_csv('ibc_new.csv')
ibc['Coincident ann'] = 100*ibc['Coincident Index'].pct_change(12)
ibc

# Set keyword ("失業" = "unemployment")
pytrends.build_payload(kw, timeframe='2004-01-01 2021-11-30', geo='JP')
gt1 = pytrends.interest_over_time()
gt1 = gt1.rename(columns = {"失業": "unemployment", "isPartial": "info"})
gt1.to_csv("gt1.csv")
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
gt1 = pd.read_csv('gt1.csv', index_col=0, date_parser=dateparse, dtype='float')

# Extract trend factor
s1 = seasonal_decompose(gt1.iloc[:,0], extrapolate_trend='freq')
t1 = s1.trend
plt.plot(t1)
plt.plot(gt1.iloc[:,0], linestyle='--')

# Check correlation
level = ibc['Coincident Index'][228:]
level.index = t1.index
cor = level.corr(t1)
print("Correlation of level: {:.2f}".format(cor))

a1 = gt1.iloc[:,0].pct_change(12)
ann = ibc['Coincident ann'][228:]
ann.index = a1.index
cor = ann.corr(a1)
print("Correlation of YoY: {:.2f}".format(cor))

# Plot trend
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t1.index, ibc['Coincident Index'][228:], linestyle='-', color='b', label='IBC')
ax.plot(t1.index, t1, linestyle='--', color='#e46409', label='google search: "unemployment"')
ax.legend()
plt.title('Google Search: "Unemployment"')
plt.savefig("google1.png")
