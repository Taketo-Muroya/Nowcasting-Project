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

kw_raw = st.sidebar.text_input('検索ワードを記入してください', '失業')
kw = ''.join(kw_raw)

st.write(f"""
### **「{kw}」** のグーグルトレンド
""")

# Set keyword ("失業" = "unemployment")
kw_list1 = [kw]
pytrends.build_payload(kw_list1, timeframe='2004-01-01 2021-11-30', geo='JP')
gt1 = pytrends.interest_over_time()

st.table(gt1.tail(10))
st.line_chart(gt1.iloc[:,0])
#gt1 = gt1.rename(columns = {"失業": "unemployment", "isPartial": "info"})
#gt1.to_csv("gt1.csv")
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#gt1 = pd.read_csv('gt1.csv', index_col=0, date_parser=dateparse, dtype='float')

#@st.cache
ibc = pd.read_csv('ibc_new.csv')
ibc['Coincident ann'] = 100*ibc['Coincident Index'].pct_change(12)
st.table(ibc.tail(10))
st.line_chart(ibc['Coincident Index'])
