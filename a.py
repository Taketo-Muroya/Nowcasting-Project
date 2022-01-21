from time import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title("Streamlit 超入門")

st.write("DataFrame")

df = pd.DataFrame({
    '１列目':[1,2,3,4],
    '２列名':[10,20,30,40]
})

st.write(df)

st.dataframe(df.style.highlight_max(axis=0), width=500, height=500)

st.table(df.style.highlight_max(axis=0))

"""
# 章
## 節
### 項

```python
import streamlit as st
import numpy as np
import pandas as pd
```

"""

df2 = pd.DataFrame(
    np.random.rand(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(df2)

st.area_chart(df2)

st.bar_chart(df2)

df3 = pd.DataFrame(
    np.random.rand(100, 2)/[50, 50] + [35.69, 139.70],
    columns=['lat', 'lon']
)

st.map(df3)

st.write('Display Image')
img = Image.open('sample.png')
st.image(img, caption='Word Cloud', use_column_width=True)

# interactive

st.write('Display Image')
if st.checkbox('Show Image'):
    img = Image.open('sample.png')
    st.image(img, caption='Word Cloud', use_column_width=True)

option = st.selectbox(
    'あなたが好きな数字を教えてください。',
    list(range(1, 11))
)

'あなたの式な数字は、', option, 'です。'

#txt = st.text_input('あなたの趣味を教えてください')
#'あなたの趣味：', txt

#con = st.slider('あなたの今の調子は？', 0, 100, 50)
#'コンディション：', con

# layout

st.sidebar.write('Side Bar')

txt = st.sidebar.text_input('あなたの趣味を教えてください')
'あなたの趣味：', txt

con = st.sidebar.slider('あなたの今の調子は？', 0, 100, 50)
'コンディション：', con

left_column, right_column = st.columns(2)
button = left_column.button('右カラムに文字を表示')
if button:
    right_column.write('ここは右カラム')

ex = st.expander('問い合わせ')
ex.write('問い合わせ内容を書く')
ex.write('問い合わせ内容を書く')
ex.write('問い合わせ内容を書く')
ex.write('問い合わせ内容を書く')

# Progress Bar

#'Start!!'

#latest_iteration = st.empty()
#bar = st.progress(0)

#import time
#for i in range(100):
#    latest_iteration.text(f'Iteration {i+1}')
#    bar.progress(i+1)
#    time.sleep(0.1)

#'Done!!'
