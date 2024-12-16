import streamlit as st
import pandas as pd

# 預設的更新日誌數據
def load_default_logs():
    return [
        {"日期": "2024-12-14", "版本": "1.0.0", "更新內容": "初始版本上線"},
        {"日期": "2024-12-15", "版本": "1.1.1", "更新內容": "導入Gemini API讓酷酷的功能更上一層樓，讓它能夠為我們提供超乎想像的服務。此外修正一些問題讓酷酷的功能更加穩定"},
        {"日期": "2024-12-16", "版本": "1.2.0", "更新內容": "新增股票討論度，來看看哪支股票被提到的最多?(雖然不一定是好事"},
        {"日期": "2024-12-16", "版本": "1.2.1", "更新內容": "修正文章太長無法透過API翻譯的問題"},

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
        st.switch_page("server.py")  # 或者執行頁面跳轉邏輯
