import os
import streamlit as st

from datamodules.LettuceDataset import LettuceDataset
from app import menu
from config import Config


if 'dataset' not in st.session_state:
    st.session_state['dataset'] = LettuceDataset(Config.TRAIN_PATH, split="train")
if 'val_preds' not in st.session_state:
    st.session_state['val_preds'] = None
    
dataset = st.session_state['dataset']
dataset.visualize_preprocess()
val_preds = st.session_state['val_preds']

menu()

title = "Yếu tố nào quyết định sinh trưởng của cây Xà Lách?"
st.markdown(f"<h1 style='text-align: center; font-size:80;'>{title}</h1><br></br>", unsafe_allow_html=True)

### NHIỆT ĐỘ
header1 = "1. Nhiệt độ là yếu tố quan trọng nhất"
st.markdown(f"<h2 style='font-size:30; color: white; background: black'>{header1}</h1>", unsafe_allow_html=True)
fig = dataset.visualize("corr", title="Bản Đồ Nhiệt các thuộc tính trong tập dữ liệu")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
            Dựa vào Bản Đồ Nhiệt, ta thấy:
            * Chỉ số **Nhiệt độ** có tương quan rõ rệt nhất (`-0.075`) với **Số ngày lớn** của cây.
            * Các yếu tố còn lại lần lượt là chỉ số **TDS (Tổng chất rắn hoà tan)**, **Độ ẩm** và cuối cùng là **Độ pH**.
            """,
            unsafe_allow_html=True)
st.markdown("""
            Ở đây, **TDS (Total Dissolved Value)** là một trong những chỉ số dùng để đo hàm lượng chất khoáng trong nước tưới cây. Chỉ số **TDS** lý tưởng cho cây sẽ nằm khoảng từ `600` đến `800`.
            """)
tds_img_path = os.path.join(Config.IMG_DIR, 'TDS_explained.jpg')
st.image(tds_img_path)

fig = dataset.visualize("bar", title="Chi tiết tương quan âm/dương của từng thuộc tính")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
            So với những yếu tố khác, tăng **Nhiệt độ** lên khả năng sẽ khiến **Số ngày cây sống** giảm xuống.

            Chỉ có duy nhất **Độ pH** có chỉ số tương quan dương, tuy nhiên cũng không cao lắm.
            """,
            unsafe_allow_html=True)

### MÔI TRƯỜNG
header2 = "2. Xà Lách ưa sống trong môi trường mát mẻ"
st.markdown(f"<h2 style='font-size:30; color: white; background: black'>{header2}</h1>", unsafe_allow_html=True)
features_to_plot = [feat for feat in dataset.df.columns if 'temperature' in feat.lower()]
fig = dataset.visualize("dist", features_to_plot, f"Phân phối của thuộc tính {features_to_plot[0]}")
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    '''
    Có khả năng các cây trong tập dữ liệu được chia làm 2 nhóm và trồng trong 2 môi trường nhiệt độ khác nhau:
    * Khoảng `18-25` độ C
    * Khoảng `29-34` độ C
    '''
)

features_to_plot = [feat for feat in dataset.df.columns if any(substring in feat.lower() for substring in ['temperature', 'day'])]
fig = dataset.visualize("scatter", features_to_plot)
st.plotly_chart(fig, use_container_width=True, title="Nhiệt độ và Số ngày cây sống")
st.markdown(
    '''
    Hầu hết xà lách được trồng nhiều và phát triển ổn định trong môi trường **từ `18` độ C đến `25` độ C**.
    Chỉ có một vài cây sống trong môi trường hơn `29` độ C. Điều này chứng tỏ cây khó phát triển trong môi trường nhiệt độ cao.
    '''
)

### YẾU TỐ KHÁC
header3 = "3. Biểu diễn phân phối của các yếu tố khác"
st.markdown(f"<h2 style='font-size:30; color: white; background: black'>{header3}</h1>", unsafe_allow_html=True)
features_to_plot = [feat for feat in dataset.df.columns if not any(substring in feat.lower() for substring in ['id', 'day', 'date'])]

fig = dataset.visualize("dist", features_to_plot)
st.plotly_chart(fig, use_container_width=True, title="Biểu đồ tần suất của các yếu tố")

### DỰ ĐOÁN
header4 = "4. Phân tích dự đoán"
st.markdown(f"<h2 style='font-size:30; color: white; background: black'>{header4}</h1>", unsafe_allow_html=True)
if st.session_state['val_preds'] is not None:
    st.markdown(
        '''
        Sử dụng **mô hình hồi quy tuyến tính đa biến**, tiến hành thẩm định mô hình trên một bộ phận dữ liệu.
        Biểu đồ 3d bên dưới thể hiện phân phối của dự đoán (màu đỏ) so với kết quả đúng (màu xanh).
        '''
    )
    x_train, x_val, y_train, y_val = dataset.preprocess()
    features_to_plot = ["Temperature (Celcius)", "TDS Value (ppm)"]
    fig = dataset.visualize_predictions(y_val, val_preds, features_to_plot, title='Biểu đồ 3D so sánh phân phối dự đoán (đỏ) và kết quả đúng (xanh) trên hai thuộc tính nhiệt độ và TDS')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '''
        Phân tích biểu đồ:
        * Có khá nhiều **các điểm xanh nằm gần với các điểm đỏ**. Điều này chứng tỏ mô hình dự đoán tốt các điểm dữ liệu đó.
        * Tuy nhiên, phân phối của **các điểm xanh nhìn chung nằm cao hơn các điểm đỏ**, đặc biệt là trong khoảng nhiệt độ `19` đến `23` độ C.
        '''
    )
    st.markdown("Có thể kết luận trong khoảng nhiệt độ `19-23` độ C, mô hình dự đoán thiếu số ngày cây lớn (under-estimate). Có thể xem xét thêm dữ liệu ở khoảng nhiệt độ này để cải thiện độ chính xác của mô hình")
    
else:
    st.markdown("Mô hình chưa được thẩm định.")
    st.markdown("Hãy chọn chức năng **Phân tích dự đoán** >> **Thẩm định** để tạo kết quả thẩm định trước.")