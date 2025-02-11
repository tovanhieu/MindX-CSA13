from datetime import datetime

import pandas as pd
import streamlit as st

from datamodules.LettuceDataset import LettuceDataset
from app import menu
from config import Config


menu()

if 'test_dataset' not in st.session_state:
    st.session_state['test_dataset'] = LettuceDataset(Config.TEST_PATH, split="test")

test_dataset = st.session_state['test_dataset']

st.header("Thêm dữ liệu mới")
plant_id = st.selectbox(
    'Chọn ID của cây',
    list(pd.unique(test_dataset.df["Plant_ID"])))
add_date = st.date_input("Ngày", datetime.today())
add_date = add_date.strftime("%m/%d/%Y")
temperature = st.number_input("Nhiệt độ (Celcius)", min_value=0, max_value=50, value=25, step=1)
humidity = st.slider("Độ ẩm (%)", min_value=0, max_value=100, value=50)
pH_level = st.number_input("Độ pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
TDS_value = st.number_input("Chỉ số TDS", min_value=0, max_value=1000, step=10)

if st.button("Thêm vào tập test", key="submit_bttn"):
    new_data = {
    'Plant_ID': plant_id,
    'Date': add_date,
    'Temperature (Celcius)': temperature,
    'Humidity (%)': humidity,
    'TDS Value (ppm)': TDS_value,
    'pH Level': pH_level,
    }
    test_dataset.append_data(new_data)
    test_dataset.write_data()
    st.write(f"Dữ liệu mới (ngày {add_date}) đã được thêm vào tập test!")
    # Reload test data
    test_dataset.read_data()