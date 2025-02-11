import pandas as pd
import streamlit as st

from datamodules.LettuceDataset import LettuceDataset
from models.LettuceGrowthModule import LettuceGrowthModule

from app import menu
from config import Config


def prettify_preds(data, preds):
    preds = pd.DataFrame(preds)
    df = data.loc[:, ["Plant_ID", "Date"]].reset_index(drop=True)
    results = pd.concat([df, preds], axis=1)
    results.columns = [*df.columns, "Predictions"]
    
    results = results.sort_values(by=["Plant_ID", "Date"])
    return round(results["Predictions"].mean())

menu()

if 'dataset' not in st.session_state:
    st.session_state['dataset'] = LettuceDataset(Config.TRAIN_PATH, split="train")
if 'test_dataset' not in st.session_state:
    st.session_state['test_dataset'] = LettuceDataset(Config.TEST_PATH, split="test")

train_dts = st.session_state['dataset']
test_dts = st.session_state['test_dataset']

x_train, x_val, y_train, y_val = train_dts.preprocess()
x_test = test_dts.preprocess()

st.header("Mô hình dự đoán")

model = st.selectbox(
    "Chọn mô hình",
    ["Linear Regression"]
)

if "module" not in st.session_state:
    st.session_state["module"] = LettuceGrowthModule(model)
module = st.session_state["module"]

if st.button("Huấn luyện (tập train)"):

    st.markdown(
        "Đang huấn luyện..."
    )
    st.session_state["trained"] = module.train(x_train, y_train)
    st.markdown(
        "Mô hình đã huấn luyện xong!"
    )

if st.button("Thẩm định (tập train)"):
    if 'trained' not in st.session_state:
        st.write("Hãy huấn luyện mô hình trước!")

    elif st.session_state["trained"] is False:
        st.write("Đã có lỗi khi huấn luyện mô hình. Kiểm tra log.")

    else:
        val_preds = module.inference(x_val)
        st.session_state['val_preds'] = val_preds
        if val_preds is not None:
            scores = module.eval(y_val, val_preds)
            st.markdown(f"""
                        Mức độ phù hợp của mô hình với tập Thẩm định là `{scores["R2 Score"]:.9f}` (R2 Score)
                        """)
        else:
            st.write("Hãy huấn luyện mô hình trước!")
    
if st.button("Kiểm tra (tập test)"):
    if 'trained' not in st.session_state:
        st.write("Hãy huấn luyện mô hình trước!")

    elif st.session_state["trained"] is False:
        st.write("Đã có lỗi khi huấn luyện mô hình. kiểm tra log.")

    else:
        test_preds = module.inference(x_test)
        if test_preds is not None:
            st.write("Số ngày trung bình cây sẽ trưởng thành:", prettify_preds(test_dts.df, test_preds))
        else:
            st.write("Hãy huấn luyện mô hình trước!")

pred_plant_id = st.selectbox(
    'Chọn ID của cây',
    list(pd.unique(test_dts.df["Plant_ID"],)),
    key="pred_plant_id")

if st.button("Dự đoán số ngày cây trưởng thành (tập test)"):
    if 'trained' not in st.session_state:
        st.write("Hãy huấn luyện mô hình trước!")

    elif st.session_state["trained"] is False:
        st.write("Đã có lỗi khi huấn luyện mô hình. kiểm tra log.")

    else:
        data = test_dts.df[test_dts.df["Plant_ID"] == pred_plant_id]
        inputs = x_test.iloc[data.index, :]
        # inputs.sort_index(inplace=True)
        preds = module.inference(inputs)
        if preds is not None:
            st.write(f"Cây {pred_plant_id} sẽ mất khoảng", prettify_preds(data, preds), "ngày để trưởng thành")
        else:
            st.write("Hãy huấn luyện mô hình trước!")
