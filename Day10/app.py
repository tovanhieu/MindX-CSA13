import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header('Mở đầu phân tích dự liệu')
st.title('Dự án phân tích dữ liệu')
st.write('Đây là mô tả cho ứng dụng phân tích dữ liệu')
name = st.text_input("Nhập tên của bạn: ")
age = st.number_input("Nhập tuổi của bạn:")
if st.button('Gửi'):
    st.write(f"Xin chào bạn {name} - {age} tuổi")
st.subheader("Hiển thị dữ liệu ngẫu nhiên")
st.subheader("Hiển thị dữ liệu quảng cáo đọc từ file")
df2 = pd.read_csv('ads_data.csv')
st.dataframe(df2)

df = pd.DataFrame(
    np.random.randn(10,3),
    columns=['Cột 1','Cột 2', 'Cột 3']
)
st.dataframe(df)

st.subheader("Hiển thị các ví dụ về biểu đồ")
st.line_chart(df)
st.bar_chart(df)    
st.area_chart(df)
st.scatter_chart(df)

st.title("Hiển thị nhiều biểu đồ trong Streamlit")

# Chia layout thành 3 cột
col1, col2, col3 = st.columns(3)
# Biểu đồ cột 1
with col1:
    st.subheader("Biểu đồ cột")
    st.bar_chart(df)
# Biểu đồ cột 2
with col2:
    st.subheader("Biểu đồ đường")
    st.line_chart(df)    
# Biểu đồ cột 3
with col3:
    st.subheader("Biểu đồ vùng")
    st.area_chart(df)  

# Vẽ và hiển thị các loại biểu đồ khác bằng matplotlib mà streamlit không hỗ trợ
st.title("Vẽ và hiển thị các loại biểu đồ khác bằng matplotlib mà streamlit không hỗ trợ")
st.header("Vẽ biểu đồ tròn")
labels = ['Python', 'HTML', 'CSS', 'JavaScript']
sizes = [40, 30, 20, 10]
colors = ['blue', 'yellow', 'pink', 'green']
fig1, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
ax.axis('equal') #Giữ hình tròn
st.pyplot(fig1)

st.header("Vẽ biểu đồ radar")
# Dữ liệu mẫu
labels = np.array(["Kỹ năng A", "Kỹ năng B", "Kỹ năng C", "Kỹ năng D", "Kỹ năng E"])
values = np.array([80, 60, 70, 90, 50])  # Điểm của từng kỹ năng
num_vars = len(labels)

# Tạo góc cho các trục radar
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values = np.concatenate((values, [values[0]]))  # Đóng vòng tròn
angles += angles[:1]  # Đóng vòng tròn

# Vẽ biểu đồ radar
fig2, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='blue', alpha=0.3)
ax.plot(angles, values, color='red', linewidth=2)
ax.set_yticklabels([])  # Ẩn nhãn trục y
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Hiển thị trong Streamlit
st.pyplot(fig2)


# Chia layout thành 3 cột
col1, col2 = st.columns(2)
with col1:
    st.subheader("Biểu đồ tròn")
    st.pyplot(fig1)
# Biểu đồ cột 2
with col2:
    st.subheader("Biểu đồ radar")
    st.pyplot(fig2)
