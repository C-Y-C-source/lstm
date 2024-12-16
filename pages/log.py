import streamlit as st
import pandas as pd

# 預設的更新日誌數據
def load_default_logs():
    return [
        {"日期": "2024-12-14", "版本": "1.0.0", "更新內容": "初始版本上線"},
        {"日期": "2024-12-15", "版本": "1.0.1", "更新內容": "修正一些問題讓酷酷的功能更加穩定"},

    ]

# 加載更新日誌
logs = pd.DataFrame(load_default_logs())

# 頁面標題
st.title("更新日誌")

all_versions = ["全部"] + logs["版本"].unique().tolist()
selected_version = st.selectbox("選擇版本", all_versions)

# 根據選擇的版本篩選日誌
if selected_version == "全部":
    filtered_logs = logs
else:
    filtered_logs = logs[logs["版本"] == selected_version]


st.dataframe(filtered_logs)

# 提供下載功能與返回主頁按鈕（同一行）
col1, col2 = st.columns([1, 1])  # 兩列均分

with col1:
    # 下載按鈕
    csv = filtered_logs.to_csv(index=False)
    st.download_button(
        label="下載 CSV",
        data=csv,
        file_name="更新日誌.csv",
        mime="text/csv",
    )

with col2:
    # 返回主頁按鈕
    if st.button("返回主頁"):
        st.experimental_rerun()  # 或者執行頁面跳轉邏輯
