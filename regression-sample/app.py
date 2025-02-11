# import os
# import streamlit as st

# from config import Config


# def menu():
#     st.sidebar.page_link("app.py", label="Home Page")
#     st.sidebar.page_link("pages/1_Data_Analysis.py", label="Phân tích tập dữ liệu")
#     st.sidebar.page_link("pages/2_Input_Record.py", label="Thêm dữ liệu dự đoán")
#     st.sidebar.page_link("pages/3_Prediction.py", label="Phân tích dự đoán")

# if __name__ == "__main__":
#     st.set_page_config(
#         page_title="Lettuce Multipage Tracker",
#         layout="centered",
#         page_icon="👋",
#     )

#     st.title("Lettuce Growth Tracker")
#     st.header("Chức năng")
#     st.markdown("""
#     1. Xem phân tích tập dữ liệu trồng Xà Lách
#     2. Thêm dữ liệu mới và cập nhật các biểu đồ
#     3. Sử dụng AI để dự đoán ngày trưởng thành của cây
#     """)
    
#     st.subheader("Credits")
#     st.markdown(

#         """
#         Ứng dựng được xây dựng với [streamlit](https://streamlit.io) và [Plotly](https://plotly.com/).
        
#         Bản quyền thuộc về CTCP Trường học MindX
#         """
#     )
#     img_path = os.path.join(Config.IMG_DIR, 'mindx_light_small.png')
#     st.image(img_path)

#     menu()

import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)