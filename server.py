
# streamlit run server.py
import streamlit as st 
# from skimage import io
# from skimage.transform import resize
import numpy as np  
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch. nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import warnings
from sklearn.model_selection import train_test_split
from streamlit_echarts import st_echarts
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import random 
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from torchtext.vocab import vocab
import re
from bs4 import BeautifulSoup
import requests
import pandas as pd
from torchtext.data.utils import get_tokenizer
import os
import pandas as pd
import torch
from collections import Counter
import streamlit as st
import joblib
from datetime import datetime
from deep_translator import GoogleTranslator

if "dialog_open" not in st.session_state:
    st.session_state.dialog_open = True  # 初始狀態為開啟

@st.dialog("請同意以下免責聲明及使用規範")
def disclaimer_dialog():
    """
    顯示免責聲明的對話框
    """
    st.markdown("""
            ## 免責聲明

            1. **預測結果僅供參考**  
            本網站提供的所有股票預測結果、數據分析及建議均僅供參考。預測結果基於歷史數據及AI模型分析，但無法保證其準確性或未來表現。使用者應自行判斷其決策並承擔風險。

            2. **不作為投資建議**  
            本網站所提供的資訊、預測及分析內容不應視為任何形式的投資建議。用戶在做出投資決策前，應諮詢專業投資顧問，並充分了解所有風險。

            3. **風險提示**  
            股票市場波動性大，投資涉及風險。過去的表現不代表未來結果，所有的投資決策和行為均由用戶自行承擔風險。網站不對任何因使用本網站資訊而產生的損失或損害負責。

            4. **數據準確性**  
            本網站提供的資料來自第三方數據源。雖然我們已竭盡全力確保資料的準確性與時效性，但無法保證所有資料無誤或無延遲。本網站對於因數據不準確或延遲造成的任何損失不承擔責任。
            
            ## 使用規範

            1. **使用者責任**  
            用戶在使用本網站服務時，應保證提供的所有資訊真實、準確且合法。用戶不得利用本網站進行任何非法活動，包括但不限於欺詐、操縱市場或違反相關法律法規。

            2. **禁止商業用途**  
            本網站僅供個人使用，不得用於任何商業用途。用戶不得將網站的內容或數據進行轉售、複製、修改或重新發布。

            3. **版權聲明**  
            本網站所有內容（包括但不限於文本、圖像、數據及預測結果）均受版權保護。未經網站明確授權，任何人不得以任何形式使用、重製或分發本網站的內容。

            4. **隱私政策**  
            我們重視用戶隱私，並承諾按照隱私政策保護用戶的個人資訊。詳細隱私政策請參見網站的隱私政策頁面。

            5. **服務中斷與修改**  
            本網站保留隨時中斷、修改或更新服務內容的權利，且不需事先通知。對於由於服務中斷或修改所導致的任何損失，我們不承擔責任。

            6. **責任限制**  
            在任何情況下，本網站對於任何因使用網站服務而產生的直接、間接、偶然、特殊或懲罰性損害概不負責。用戶使用本網站服務的風險由用戶自行承擔。
    """)
    if st.button("我已閱讀並同意"):
        st.session_state.dialog_open = False  # 標記對話框應關閉
        st.rerun()  # 強制重新執行以移除對話框


    
# 模型載入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@st.cache_resource 
def load_model_1():#lstm_1
    
    class LSTM_1(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM_1, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.1) 

        def forward(self, x):
        
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0)) 
            out = self.dropout(out)
            out = self.fc(out[:, -1, :])
            return out
        
    input_dim = 5
    hidden_dim =64
    num_layers = 2
    output_dim = 1
    weightpath='LSTM.pth'
    state_dict = torch.load(weightpath, map_location=device)
    model_1 = LSTM_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model_1.load_state_dict(state_dict) 
    model_1 = model_1.to(device)
    model_1.eval()
    return model_1

model_1 = load_model_1()
one_time=0



@st.cache_resource
def load_and_process_data(file_path, min_freq=15):
    """
    加載數據並進行分詞和詞彙表建立。
    此函數僅運行一次，返回分詞器和詞彙表。
    """
    # 加載 CSV 文件
    df = pd.read_csv(file_path)
    reviews = df['review'].values
    sentiments = df['sentiment'].values

    # 初始化分詞器
    tokenizer = get_tokenizer('basic_english')

    # 計數單詞頻率
    counter = Counter()
    for review in reviews:
        token = tokenizer(review)
        counter.update(token)

    # 建立詞彙表
    token_vocab = vocab(counter, min_freq=min_freq, specials=('<pad>', '<unk>'))
    token_vocab.set_default_index(token_vocab.get_stoi()['<unk>'])

    return tokenizer, token_vocab, reviews, sentiments


# 加載數據和詞彙表
file_path = 'IMDB_Dataset.csv'
tokenizer, token_vocab, reviews, sentiments = load_and_process_data(file_path)
PAD_IDX = token_vocab.get_stoi()['<pad>']
INPUT_DIM = len(token_vocab)


label_decoding = {0:'negative', 1:'positive'}
def predict_sentiment(text):
    # 自動轉義單引號


    # 步驟 1：分詞和編碼
    tokens = tokenizer(text)  # 使用分詞器將文本分詞
    indices = token_vocab.lookup_indices(tokens)  # 將分詞結果轉換為數字索引
    text_tensor = torch.tensor(indices).unsqueeze(0).to(device)  # 將數字索引轉換為Tensor並加入batch維度

    # 步驟 2：使用模型進行預測
    model_2.eval()  # 設置模型為評估模式
    with torch.no_grad():  # 禁用梯度計算
        output = model_2(text_tensor)  # 模型預測
        prediction = torch.sigmoid(output).item()  # 取出預測結果並應用sigmoid函數
    pred = (output.view(-1) > 0.5)
    return label_decoding[int(pred)]
    #print('Pred Label:',label_decoding[int(pred)])             # 顯示文字 

    # 示例輸入

def load_model_2():#lstm_2
    
    class IMDB(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
        def __getitem__(self, index):
            return self.x[index], self.y[index]
        
        def __len__(self):
            return len(self.x)

    class LSTM_2(nn.Module):
        def __init__(self, embedding_dim, hidden_size, num_layers=1, bidirectional=True):
            super().__init__()
            self.embedding = nn.Embedding(INPUT_DIM,  embedding_dim, padding_idx = PAD_IDX)
            self.lstm =nn.LSTM(embedding_dim, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            batch_first=True
            )

            hidden = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Linear(hidden, 1)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            emb_out = self.embedding(x)
            out, (h, c)  = self.lstm(emb_out)
            x = out[:, -1, :]
            x = self.fc(x)
            print(x.shape)
            return self.sigmoid(x)
        
        
    model_2 = LSTM_2(embedding_dim = 300, hidden_size= 512).to(device)

    model_path='mode5l.pt'

    model_2.load_state_dict(torch.load(model_path, map_location=device))
    model_2.eval()
    return model_2

model_2 = load_model_2()

def cnn2dm():
    timestep = 20  # 時間步長
    feature_size = 5  # 特徵數量
    out_channels = [16, 32, 64]  # 卷積輸出通道
    output_size = 1  # 輸出大小
    class CNN2D(nn.Module):
        def __init__(self, feature_size, timestep, out_channels, output_size):
            super(CNN2D, self).__init__()

            # 定义二维卷积层
            self.conv2d_1 = nn.Conv2d(1, out_channels[0], kernel_size=3, padding=1)
            self.conv2d_2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, padding=1)
            self.conv2d_3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1)

            # Pooling layer to reduce dimensions
            self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

            # Calculate output size after convolution and pooling
            self.conv_output_size = self.calculate_conv_output_size(timestep, feature_size)

            # Define fully connected layers
            self.fc_1 = nn.Linear(896, 256)
            self.fc_2 = nn.Linear(256, output_size)

            # Activation function
            self.relu = nn.ReLU()

        def calculate_conv_output_size(self, timestep, feature_size):
            def calc_dim(input_size, kernel_size, stride, padding):
                return (input_size - kernel_size + 2 * padding) // stride + 1

            # First convolution layer
            height = calc_dim(timestep, 3, 1, 1)
            width = calc_dim(feature_size, 3, 1, 1)

            # First pooling layer
            height = calc_dim(height, 2, 2, 0)
            width = calc_dim(width, 2, 2, 0)

            # Second convolution layer
            height = calc_dim(height, 3, 1, 1)
            width = calc_dim(width, 3, 1, 1)

            # Second pooling layer
            height = calc_dim(height, 2, 2, 0)
            width = calc_dim(width, 2, 2, 0)

            # Third convolution layer
            height = calc_dim(height, 3, 1, 1)
            width = calc_dim(width, 3, 1, 1)

            # Third pooling layer
            height = calc_dim(height, 2, 2, 0)
            width = calc_dim(width, 2, 2, 0)

            return height, width
        def forward(self, x):
            #print(f"Input shape: {x.shape}")  # Debug
            x = self.pool(self.relu(self.conv2d_1(x)))
            #print(f"After conv2d_1: {x.shape}")  # Debug
            x = self.pool(self.relu(self.conv2d_2(x)))
            #print(f"After conv2d_2: {x.shape}")  # Debug
            x = self.pool(self.relu(self.conv2d_3(x)))
            #print(f"After conv2d_3: {x.shape}")  # Debug
            x = x.view(x.size(0), -1)
            #print(f"Flattened shape: {x.shape}")  # Debug
            x = self.relu(self.fc_1(x))
            x = self.fc_2(x)
            return x
    
    model_3 = CNN2D(feature_size=feature_size, timestep=timestep, out_channels=out_channels, output_size=output_size)
    model_3.load_state_dict(torch.load("2dcnn.pth"))
    timestep = 10  # 时间步，与训练时一致
    model_3.eval()  # 确保模型处于评估模式
    return model_3

model_dcnn = cnn2dm()
# 主邏輯
if st.session_state.dialog_open:
    disclaimer_dialog()
    


import google.generativeai as genai
import streamlit as st

# 設置標題
st.title("有問題嗎?問問Gemini")
st.caption("🚀power by gemini")
with st.expander("如何申請 Gemini API 密鑰"):
    st.write("""
        要在你的應用程式中使用 Gemini API，你需要一個 API 密鑰。請依照以下步驟申請密鑰：

        1. **註冊 Google Cloud 帳號：**
           - 造訪 [Google Cloud 官方網站](https://cloud.google.com/)，並註冊一個帳號。如果你已經有帳號，請直接登入。

        2. **啟用 Gemini API：**
           - 登入後，前往 [API 服務頁面](https://console.cloud.google.com/).
           - 搜尋 **"Gemini"** 並啟用 Gemini API。
           - 點擊 **"啟用"** 來開通該服務。

        3. **取得 API 密鑰：**
           - 在 API 設定頁面，進入 **Credentials**（認證）選項。
           - 點擊 **"Create Credentials"**（創建認證），選擇 **API key**。
           - 系統將生成一個 API 密鑰，記得複製並妥善保存。

        4. **將 API 密鑰添加到你的應用程式中：**
           - 現在你可以將這個 API 密鑰貼入到你的應用程式中，例如在你的 Streamlit 應用中。
           - 確保密鑰以 **"AIza"** 開頭（例如："AXXXXXXXXXXXXXXX--6XXXXXXXXXXXXXXXXXX"）。

        5. **安全性注意事項：**
           - 請保管好你的 API 密鑰，避免公開分享或暴露在客戶端程式碼中。
           - 建議使用環境變數或密鑰管理工具來提升安全性。

        取得 API 密鑰後，你就可以將其整合到你的應用程式中，使用 Gemini API 提供的功能來生成內容。
    """)
# Add custom CSS to adjust the position of the chat input box
st.markdown(
    """
    <style>
    .stTextInput > div {
        margin-bottom: 400px;  /* Adjust this value to move the input box up */
    }
    </style>
    """,
    unsafe_allow_html=True
)

api_key = "AIzaSyD2RxE2U6ZgJJHJHXcqOpXiG_--TQxvhCI"
genai.configure(api_key=api_key)
model_name = st.selectbox(
    "請選擇Gemini模型",
    options=["gemini-2.0-flash-exp","gemini-2.0-flash-lite","gemini-2.0-flash-thinking-exp-01-21"],
    key="model_selector"
)

st.caption(f"當前選擇的模型是: {model_name}")

model = genai.GenerativeModel(model_name)

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("LSTM是甚麼? 問問Gemini"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = model.generate_content(prompt)
            reply = response.text
            st.markdown(reply)
        except Exception as e:
            reply = f"Error: {e}"
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.write("---")
with st.sidebar:
    st.header("聯絡資訊")
    st.write("如果您有任何問題，請隨時聯繫我們！")

    # 聯絡資訊
    st.subheader("聯絡方式")
    st.write("電子郵件: 411123002@gms.ndhu.edu.tw")

    
    st.write("---")
    
    # 其他側邊欄內容
    st.subheader("網站導航")
    st.write("[首頁](https://stocksupporter.streamlit.app/)")
    st.write("[Github](https://github.com/kkk-source)")

    st.write("[隱私政策](/privacy_policy)")
    st.write("[更新日誌](/log)")
    st.write("---")
    
    # 顯示版權資訊
    st.text("© 2024 保留所有權利。")


st.title("台灣加權指數(^TWII)")
st.caption("備註：數據來源為 Yahoo Finance，更新間隔為 1 分鐘。且不包含盤後數據。")

import yfinance as yf
import streamlit as st
stock_code = "^TWII"
stock = yf.Ticker(stock_code)
data_today = stock.history(period="1d", interval="1m")  
data_recent = stock.history(period="5d", interval="1d") #
if not data_today.empty and len(data_recent) >= 2:
    # 今日開盤價、最高價、最低價
    today_open = data_today["Close"].iloc[-1]
    today_high = data_today["High"].max()
    today_low = data_today["Low"].min()  
    
    # 前一天收盤價
    yesterday_close = data_recent["Close"].iloc[-2]
    yesterday_h = data_today["High"].iloc[-2]
    yesterday_l = data_today["Low"].iloc[-2]
    # 計算差異
    open_diff = today_open - yesterday_close
    high_diff = today_high - yesterday_h
    low_diff = today_low - yesterday_l
    open_diff = today_open - yesterday_close
    high_diff = today_high - yesterday_h
    low_diff = today_low - yesterday_l
    
    open_pct = (open_diff / yesterday_close) * 100
    high_pct = (high_diff / yesterday_h) * 100
    low_pct = (low_diff / yesterday_l) * 100
    col31, col32, col33 = st.columns(3)
    col31.metric("時實數據", f"{today_open:.2f}", f"{open_diff:+.2f} ({open_pct:+.2f}%)")
    col32.metric("今日最高", f"{today_high:.2f}", f"{high_diff:+.2f} ({high_pct:+.2f}%)")
    col33.metric("今日最低", f"{today_low:.2f}", f"{low_diff:+.2f} ({low_pct:+.2f}%)")
else:
    st.error("無法獲取完整的數據，請稍後再試。")
lstm_intro = """
### 什麼是 LSTM（長短期記憶）？

LSTM（Long Short-Term Memory）是一種特殊的循環神經網絡（RNN），它能夠在長時間內保持記憶，從而解決傳統 RNN 在處理長序列時的梯度消失問題。

#### LSTM 的結構
LSTM 由以下幾個主要部分組成：

1. **遺忘門（Forget Gate）：**
   - 決定哪些信息應該被丟棄，哪些應該保留。它檢查前一狀態的輸出和當前的輸入，並輸出一個 0 到 1 之間的值，表示應保留的記憶。

2. **輸入門（Input Gate）：**
   - 決定當前輸入應該對記憶進行多少修改。它包含兩個部分：一個是用來更新記憶的候選層，另一個是控制有多少候選記憶應該被加入到單元狀態中的部分。

3. **單元狀態（Cell State）：**
   - 存儲了過去的長期記憶，並根據忘記門和輸入門的結果進行更新。這是 LSTM 的關鍵部分。

4. **輸出門（Output Gate）：**
   - 根據單元狀態和當前輸入，決定輸出多少信息到下個時間步。

#### LSTM 的優點
- **解決梯度消失問題：** 相比於傳統的 RNN，LSTM 可以捕捉長期依賴，並且能夠防止梯度消失問題。
- **時間序列預測：** LSTM 特別適合處理時間序列數據，比如語音識別、語言建模等。

#### LSTM 的應用領域
- **語音識別：** 用於將語音信號轉換為文字。
- **語言處理：** 用於機器翻譯和情感分析等任務。
- **金融預測：** 用於股票價格預測、銷售預測等。

LSTM 是處理時間序列數據的強大工具，能夠記住關鍵的時間步信息，並忽略不必要的噪聲。
"""
cnn_intro = """
### 什麼是 2D CNN（卷積神經網絡）？

2D 卷積神經網絡（2D CNN）是一種深度學習模型，主要用於處理2D數據（如圖像）。它通過卷積操作學習圖像中的空間特徵，並通過多層堆疊來提取從低層到高層的特徵。

#### 2D CNN 的結構
1. **卷積層（Convolutional Layer）：**
   - 使用濾波器（或稱為卷積核）來掃描圖像，提取圖像的局部特徵（如邊緣、角落等）。這一層通過卷積運算生成特徵圖（feature map）。

2. **池化層（Pooling Layer）：**
   - 用於縮小圖像的空間尺寸，從而減少計算量並防止過擬合。最常見的是最大池化（Max Pooling）和平均池化（Average Pooling）。

3. **激活函數（Activation Function）：**
   - 通常使用 ReLU（Rectified Linear Unit）激活函數，來增加非線性，使神經網絡能夠學習更加複雜的模式。

4. **全連接層（Fully Connected Layer）：**
   - 在卷積層和池化層之後，將學到的特徵進行分類或回歸任務。

5. **輸出層（Output Layer）：**
   - 根據任務的需求，輸出不同的結果，分類任務通常使用 softmax 函數進行多分類，回歸任務則直接輸出連續值。

#### 2D CNN 的優點
- **空間不變性：** CNN 能夠學習圖像中的局部特徵，並對圖像進行平移、旋轉等變換的判斷。
- **參數共享：** 卷積層中的濾波器是共享的，這意味著每個濾波器在整個圖像中都是相同的，這大大減少了參數數量。
- **層次特徵學習：** 通過多層卷積和池化操作，CNN 能夠從低層到高層逐步學習圖像中的複雜特徵。

#### 2D CNN 的應用領域
- **圖像分類：** 用於對不同類別的圖像進行分類（例如，辨識貓狗圖像）。
- **物體檢測：** 用於識別圖像中的特定物體位置。
- **面部識別：** 用於檢測圖像中的人臉並進行識別。
- **醫學影像分析：** 用於分析醫學影像（如 X 光片、CT 扫描等）進行疾病診斷。

2D CNN 在圖像處理領域取得了顯著的成功，並成為許多視覺識別任務的基礎。
"""


import streamlit as st
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

def csv_content():
    base_url = "https://www.ptt.cc/bbs/Stock/index.html"
    posts = []
    min_length = 50  # 最小內文長度限制
    target_count = 60  # 目標文章數量
    progress = 0  # 初始進度

    # Streamlit 進度條
    progress_bar = st.progress(progress)
    status_text = st.empty()

    while len(posts) < target_count:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 獲取符合條件的文章
            new_posts = get_ptt_posts(soup, min_length)
            posts.extend(new_posts)
            
            # 更新進度條
            progress = min(len(posts) / target_count, 1.0)  # 確保進度不超過 100%
            progress_bar.progress(progress)
            status_text.text(f"已抓取文章數量：{len(posts)}/{target_count}")
            deletedcontain=len(posts)-target_count
            

            # 如果總數已達目標數量，則結束
            if len(posts) >= target_count:
                posts = posts[:target_count]  # 只保留目標數量的文章
                break

            # 找到上一頁的連結
            prev_link_tag = soup.find("a", class_="btn wide", string="‹ 上頁")
            if prev_link_tag:
                prev_link = "https://www.ptt.cc" + prev_link_tag["href"]
                base_url = prev_link
            else:
                st.error("無法找到上一頁，停止爬取")
                break
        else:
            st.error(f"無法取得完整網頁內容，HTTP 狀態碼：{response.status_code}")
            break

        time.sleep(0.5)  # 避免過於頻繁的請求

    # 匯出資料
    if posts:
        df = pd.DataFrame(posts)
        df.to_csv(r"ptt_stock_filtered_content.csv", index=False, encoding="utf-8-sig")
        

    else:
        st.error("沒有抓取到任何文章")
    return len(posts),deletedcontain

def get_ptt_posts(soup, min_length):
    """從 PTT 頁面解析符合條件的文章"""
    data = soup.select("div.r-ent")
    result = []
    for item in data:
        try:
            # 抓取文章標題
            title = item.select_one("div.title").text.strip()
            if "[公告]" in title:
                continue
            # 抓取文章連結
            link_tag = item.select_one("div.title a")
            if link_tag:
                article_link = "https://www.ptt.cc" + link_tag["href"]
                article_response = requests.get(article_link)
                if article_response.status_code == 200:
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    # 抓取文章完整內容
                    full_content = article_soup.select_one("div#main-content").text.strip()
                    # 使用正則表達式提取正文內容
                    content_match = re.search(r"時間.*?\n(.*?)(?:--|$)", full_content, re.DOTALL)
                    content = content_match.group(1).strip() if content_match else "無法提取內文"
                else:
                    content = "無法取得內容"
            else:
                content = "無法取得內容"
            
            # 如果內文長度小於指定最小字數，則跳過
            if len(content) >= min_length and  len(content)<2000:
                result.append({"title": title, "content": content})
        except Exception as e:
            print(f"發生錯誤：{e}")
            continue
    return result

def sen_ana(sentiment_counts):
    file_path=r'ptt_stock_filtered_content.csv'
    data = pd.read_csv(file_path)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_items = len(data['content'])
    for index, content in enumerate(data['content']):
        if isinstance(content, str):

            content = content.replace("\r\n", " ").replace("\n", " ").replace("　", " " ).replace("-", " ").strip()
            content = ' '.join(content.split())
            content = re.sub(r'[^\w\s]', '', content)
            content = content.lower()
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
            #
            if(((index + 1) / total_items)==50):
                st.caption(content)
                st.caption(len(content))
            
            translation = GoogleTranslator(source='zh-TW', target='en').translate(content)

            sentiment_code = predict_sentiment(translation)
            

            
            sentiment_label = "positive" if sentiment_code == "positive" else "negative"

            sentiment_counts[sentiment_label] += 1
            
            progress_bar.progress((index + 1) / total_items)
            status_text.text(f"已完成數量：{index + 1}/{60}")

            results.append({"Original": content, "Translated": translation, "Sentiment": sentiment_label})
            
    return results

def display_bar_chart(sentiment_counts):
    # 配置直方圖選項
    option = {
        "backgroundColor": "#212121",
        "title": {
            "text": "情感分析統計",
            "subtext": "資料來源:股票論壇",
            "x": "left",
            "textStyle": {
                "color": "#f2f2f2"
            }
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow"
            }
        },
        "legend": {
            "data": ["COUNT"],
            "textStyle": {
                "color": "#f2f2f2"
            }
        },
        "xAxis": {
            "type": "category",
            "data": ["positive","negative"],
            "axisLine": {
                "lineStyle": {
                    "color": "#f2f2f2"
                }
            },
            "axisLabel": {
 
                "interval": 0 
            }
        },
        "yAxis": {
            "type": "value",
            "axisLine": {
                "lineStyle": {
                    "color": "#f2f2f2"
                }
            }
        },
        "series": [
            {
                "name": "COUNT",
                "type": "bar",
                "data": [sentiment_counts['positive'],sentiment_counts['negative']],
                "itemStyle": {
                    "color": "#ef4136"
                }
            }
        ]
    }

    st_echarts(options=option, height="600px")
 

# --------------------------------------------------------------------------------------
st.write("---")
def load_and_process_data(file_path):
    # 定義熱門股票代碼與公司名稱（這部分保持不變）
    popular_stocks = {
        "2618": "長榮航",
        "2883": "凱基金",
        "9105": "泰金寶-DR",
        "2303": "聯電",
        "3706": "神達",
        "3481": "群創",
        "3019": "亞光",
        "00919": "群益台灣精選高息",
        "2888": "新光金",
        "2887": "台新金",
        "2354": "鴻準",
        "2317": "鴻海",
        "6770": "力積電",
        "2353": "宏碁",
        "3029": "零壹",
        "2313": "華通",
        "2609": "陽明",
        "00965": "元大航太防衛科技",
        "2344": "華邦電",
        "3037": "欣興",
        "4540": "全球傳動",
        "2002": "中鋼",
        "2368": "金像電",
        "2890": "永豐金",
        "2312": "金寶",
        "2886": "兆豐金",
        "2308": "台達電",
        "00637L": "元大滬深300正2",
        "00939": "統一台灣高息動能",
        "1605": "華新",
        "5876": "上海商銀",
        "2409": "友達",
        "00934": "中信成長高股息",
        "3231": "緯創",
        "3450": "聯鈞",
        "00945B": "凱基美國非投等債",
        "2383": "台光電",
        "2891": "中信金",
        "2880": "華南金",
        "00918": "大華優利高填息30",
        "2892": "第一金",
        "4906": "正文",
        "2359": "所羅門",
        "2356": "英業達",
        "2392": "正崴",
        "2027": "大成鋼",
        "2603": "長榮",
        "1314": "中石化",
        "2399": "映泰",
        "00665L": "富邦恒生國企正2",
        "1101": "台泥",
        "2884": "玉山金",
        "1326": "台化",
        "2606": "裕民",
        "8070": "長華",
        "1402": "遠東新",
        "2455": "全新",
        "2382": "廣達",
        "6505": "台塑化",
        "00753L": "中信中國50正2",
        "8046": "南電",
        "00929": "復華台灣科技優息",
        "3032": "偉訓",
        "6116": "彩晶",
        "00830": "國泰費城半導體",
        "2371": "大同",
        "3014": "聯陽",
        "5880": "合庫金",
        "1528": "恩德",
        "5871": "中租-KY",
        "2454": "聯發科",
        "2834": "臺企銀",
        "3045": "台灣大",
        "00680L": "元大美債20正2",
        "00715L": "期街口布蘭特正2",
        "00940": "元大台灣價值高息",
        "1808": "潤隆",
        "2324": "仁寶",
        "1504": "東元",
        "2867": "三商壽",
        "3090": "日電貿",
        "3062": "建漢",
        "9802": "鈺齊-KY",
        "00915": "凱基優選高股息",
        "3661": "世芯-KY",
        "0056": "元大高股息",
        "4977": "眾達-KY",
        "2301": "光寶科",
        "1514": "亞力",
        "00882": "中信中國高股息",
        "1904": "正隆",
        "2637": "慧洋-KY",
        "5284": "jpp-KY",
        "2412": "中華電",
        "3711": "日月光投控",
        "6005": "群益證",
        "00938": "凱基優選30",
        "1316": "上曜",
        "2889": "國票金",
        "1513": "中興電"
    }
   # 創建一個反向映射
    stock_name_to_code = {name: code for code, name in popular_stocks.items()}

    # 合併所有股票代碼與名稱為單個正則表達式
    all_patterns = "|".join([fr"\b{re.escape(code)}\b" for code in popular_stocks.keys()] +
                            [re.escape(name) for name in popular_stocks.values()])
    regex = re.compile(all_patterns)

    # 讀取 CSV 檔案
    df = pd.read_csv(file_path)

    # 提取股票函數
    def extract_stocks(content, regex, stock_dict, name_to_code_dict):
        matches = regex.findall(str(content))
        if matches:
            unified_matches = set()
            for match in matches:
                if match in stock_dict:
                    unified_matches.add(stock_dict[match])
                elif match in name_to_code_dict:
                    unified_matches.add(match)
            return ", ".join(unified_matches) if unified_matches else ""
        return ""

    # 在 content 欄位中進行提取
    if "content" in df.columns:
        df["相關股票"] = df["content"].apply(lambda x: extract_stocks(x, regex, popular_stocks, stock_name_to_code))

    # 篩選出包含股票的文章
    result = df[df["相關股票"] != ""]

    # 統計股票被提及次數
    stock_counts = result["相關股票"].str.split(", ").explode().value_counts()

    # 將股票代碼替換回股票名稱
    stock_counts.index = stock_counts.index.map(lambda x: popular_stocks.get(x, x))

    return stock_counts
def main2():
    st.title("股票論壇討論熱度")
    st.caption("")
    # 讀取並處理資料
    file_path = "ptt_stock_filtered_content.csv"
    stock_counts = load_and_process_data(file_path)

    stock_counts_sorted = stock_counts.sort_values(ascending=True)
    
    # 將排序後的結果放入 DataFrame，並重新排序索引
    bottom_10_df = stock_counts_sorted.head(60).reset_index()
    bottom_10_df.columns = ['股票', '討論次數']

    # 重新排序後設置索引，保證視覺化順序正確
    bottom_10_df = bottom_10_df.sort_values(by='討論次數', ascending=True)

    # 使用 Streamlit 的 bar_chart 正確顯示
    st.bar_chart(bottom_10_df.set_index('股票')['討論次數'])

main2()

st.title("市場情感判斷")
sentiment_counts = {"positive": 0, "negative": 0}
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        clicked1 = st.button("更新文章", help="更新最近最多 60 篇有效文章")
        if clicked1:
            len_post,deletedcontain=csv_content()
    
    with col2:
        clicked2 = st.button("更新情感統計", help="更新最近最多 60 篇有效文章情感統計")
        if clicked2:
            
            results=sen_ana(sentiment_counts)
            if sentiment_counts['positive']>sentiment_counts['negative']:
                st.balloons()
            else:
                st.snow()
                        #len_post,deletedcontain=csv_content()
    
colsuccess, colwarning = st.columns(2)
if clicked1:
    with colsuccess:
        st.success(f"資料已成功匯出點擊更新情感以更新有效文章情感統計：{len_post}")


if clicked2:
    sentiment_df = pd.DataFrame(results)
    st.write("### Sentiment Counts:", sentiment_counts)
    positive_count = sentiment_counts['positive']
    negative_count = sentiment_counts['negative']
    display_bar_chart(sentiment_counts)
    sentiment_df = pd.DataFrame(results)
    st.markdown("### Results Table")
    st.write(sentiment_df)
# 雙向 LSTM 介紹
with st.expander("雙向 LSTM (BiLSTM) 介紹"):
    st.markdown("""
    ## 雙向 LSTM (BiLSTM) 介紹

    雙向 LSTM（Bidirectional Long Short-Term Memory）是一種特別的 LSTM 模型，它在處理序列數據時，將輸入序列同時從兩個方向進行處理——正向和反向。這使得模型能夠利用更多上下文信息，從而提高預測性能。

    ### BiLSTM 的結構

    與傳統的 LSTM 模型相比，BiLSTM 擁有兩層 LSTM 組件：
    1. **正向 LSTM**：從序列的開始處到結束，順序地處理數據。
    2. **反向 LSTM**：從序列的結束處回到開始，逆向處理數據。

    這兩層 LSTM 的輸出會被結合（通常是串接或加權平均），形成最終的輸出。這樣做的目的是讓模型能夠同時考慮序列的過去和未來信息。

    ### BiLSTM 的應用

    BiLSTM 主要應用於需要上下文信息的序列處理任務：
    - **語言模型**：語言理解、情感分析。
    - **語音識別**：可以考慮語音的上下文。
    - **機器翻譯**：處理源語言和目標語言的上下文信息。

    ### 優點
    - 能夠捕捉更多的上下文信息（雙向的前後關係）。
    - 增加了模型的表達能力，適用於更復雜的序列數據。

    ### 缺點
    - 計算量和內存需求較高，因為模型需要處理兩個方向的序列。
    """)

# 注意力機制介紹
with st.expander("注意力機制 (Attention Mechanism) 介紹"):
    st.markdown("""
    ## 注意力機制 (Attention Mechanism) 介紹

    注意力機制是一種模仿人類視覺注意力的算法，用來使模型能夠專注於序列中的關鍵部分。它在處理長序列時尤其有用，因為它可以幫助模型“選擇性地”關注序列中的重要位置，而非對整個序列進行平等的處理。

    ### 注意力機制的工作原理

    注意力機制會根據某個輸入的“查詢”來計算每個元素的權重，這些權重決定了模型應該將多少注意力集中在該元素上。通常，這些權重是通過計算查詢與所有鍵的相似度來獲得的。

    - **查詢 (Query)**：用來查找序列中相關信息的向量。
    - **鍵 (Key)**：序列中的每個元素，模型用它來決定是否需要關注該元素。
    - **值 (Value)**：對應於鍵的輸出，經過加權後被選擇性地用於最終輸出。

    在計算過程中，通過內積或其他相似度測量來計算查詢和鍵之間的相似度，然後根據相似度為值分配權重。

    ### 注意力機制的應用

    - **機器翻譯**：注意力機制可以讓模型在生成翻譯時專注於源語言的關鍵部分。
    - **圖像描述生成**：在生成描述時，模型可以專注於圖像中的重要區域。
    - **語音識別**：在語音轉文字的過程中，模型可以選擇性地專注於特定的時間步。

    ### 優點
    - 能夠處理長序列數據，克服了傳統 RNN 和 LSTM 在長距離依賴處理上的局限性。
    - 增強了模型的可解釋性，可以清楚地看到模型關注的關鍵部分。

    ### 缺點
    - 計算開銷較大，尤其是在長序列的情況下。
    - 需要更多的計算資源，尤其是在多層注意力結構中。
    """)

# --------------------------------------------------------------------------------------

st.write("---")
st.markdown("# 算法股價建議價格")
st.markdown("### 深度卷積網路(2D CNN)")

timestep = 10


# 加载Scaler
sc = joblib.load("sc.pkl")
sc_2 = joblib.load("sc_2.pkl")

ticker_all = ["2303.TW", "2330.TW", "2317.TW", "2412.TW"]
pre_all = [0] * len(ticker_all)
compare_dcnn_data = [0] * len(ticker_all)
open_diff = [0] * len(ticker_all)
open_pct = [0] * len(ticker_all)

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = '2024-11-28'

for i, ticker_ever in enumerate(ticker_all):
    data = yf.download(ticker_ever, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    data_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[data_columns].values.astype('float')

    compare_dcnn_data[i] = data[-1, 0]

    data = sc.transform(data)

    def prepare_prediction_data(data, timestep):
        input_data = []
        for j in range(len(data) - timestep):
            input_data.append(data[j:j + timestep])
        return np.array(input_data)
    
    input_data = prepare_prediction_data(data, timestep)
    input_tensor = torch.from_numpy(input_data).to(torch.float32)

    with torch.no_grad():
        predictions = model_dcnn(input_tensor.unsqueeze(1))  # 添加通道维度
    predictions = predictions.numpy()
    predictions_inverse = sc_2.inverse_transform(predictions)
    

    pre_all[i] = round(predictions_inverse[-1].item(), 1)
    open_diff[i] = pre_all[i]-compare_dcnn_data[i]
    open_pct[i] = (open_diff[i] / compare_dcnn_data[i]) * 100

with st.container():
    columns = st.columns(len(ticker_all))
    
    for i, ticker_ever in enumerate(ticker_all):
        columns[i].metric(
            label=ticker_ever,
            value=f"{pre_all[i]}",
            delta=f"{open_diff[i]:+.2f} ({open_pct[i]:+.2f}%)"
        )
with st.expander("點擊查看 2D CNN 介紹"):
    st.markdown(cnn_intro)       
# -------------------------------
st.markdown("### 長短效神經網路(LSTM)")    
def transform_data(df):
    data_index =  ['Open','High','Low','Close','Volume']
    flatten_data = df[data_index].values.reshape(-1)  # 攤平資料
    str_data = "<SEP>".join(flatten_data.astype('str'))
    filter_data = str_data.replace(',',"").replace('X',"")
    x_data = filter_data.split("<SEP>")
    return x_data


for i, ticker_ever in enumerate(ticker_all):
    data = yf.download(ticker_ever, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    data_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[data_columns].values.astype('float')

    compare_dcnn_data[i] = data[-1, 0]

    data = sc.transform(data)

    def prepare_prediction_data(data, timestep):
        input_data = []
        for j in range(len(data) - timestep):
            input_data.append(data[j:j + timestep])
        return np.array(input_data)
    
    input_data = prepare_prediction_data(data, timestep)
    input_tensor = torch.from_numpy(input_data).to(torch.float32)

    with torch.no_grad():
        predictions = model_1(input_tensor)  
    predictions = predictions.numpy()
    predictions_inverse = sc_2.inverse_transform(predictions)
    

    pre_all[i] = round(predictions_inverse[-1].item(), 1)
    open_diff[i] = pre_all[i]-compare_dcnn_data[i]
    open_pct[i] = (open_diff[i] / compare_dcnn_data[i]) * 100


with st.container():
    columns1 = st.columns(len(ticker_all))
    
    for i, ticker_ever in enumerate(ticker_all):
        columns1[i].metric(
            label=ticker_ever,
            value=f"{pre_all[i]}",
            delta=f"{open_diff[i]:+.2f} ({open_pct[i]:+.2f}%)"
        )

with st.expander("點擊查看 LSTM 介紹"):
    st.markdown(lstm_intro)
# --------------------------------------------------------------------------------------
