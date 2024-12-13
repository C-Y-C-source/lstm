
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
import pyautogui
import joblib
from datetime import datetime

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
        
    input_dim = 7
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

# 主邏輯
if st.session_state.dialog_open:
    disclaimer_dialog()






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

    

with st.sidebar:
    st.header("聯絡資訊")
    st.write("如果您有任何問題，請隨時聯繫我們！")

    # 聯絡資訊
    st.subheader("聯絡方式")
    st.write("電子郵件: 411123002@gms.ndhu.edu.tw")
    st.write("電話: 0800-000-050")
    
    st.write("---")
    
    # 其他側邊欄內容
    st.subheader("網站導航")
    st.write("[首頁](https://vskuzygkniaubvpqqbzoot.streamlit.app/)")
    st.write("[Github](https://github.com/kkk-source)")
    st.write("[FAQ](https://github.com/kkk-source/lstm/issues/1)")
    st.write("[隱私政策](/privacy_policy)")
    
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



import streamlit as st
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

def csv_content():
    """爬取 PTT 股票板最近 60 篇有效文章並儲存為 CSV"""
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
            st.error(f"無法取得網頁內容，HTTP 狀態碼：{response.status_code}")
            break

        time.sleep(1)  # 避免過於頻繁的請求

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
            if len(content) >= min_length:
                result.append({"title": title, "content": content})
        except Exception as e:
            print(f"發生錯誤：{e}")
            continue
    return result

# --------------------------------------------------------------------------------------
st.write("---")
st.title("市場情感判斷")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        clicked1 = st.button("更新文章", help="更新最近 60 篇有效文章")
        if clicked1:
            len_post,deletedcontain=csv_content()
    
    with col2:
        clicked2 = st.button("更新情感統計", help="更新最近 60 篇有效文章情感統計")
        if clicked2:
            len_post,deletedcontain=csv_content()
colsuccess, colwarning = st.columns(2)
if clicked1:
    with colsuccess:
        st.success(f"資料已成功匯出點擊更新情感以更新有效文章情感統計：{len_post}")
    with colwarning:
        st.warning(f"已刪除{deletedcontain}篇不合要求之文章")


# --------------------------------------------------------------------------------------
st.write("---")
timestep = 10


# 加载Scaler
sc = joblib.load("sc.pkl")
sc_2 = joblib.load("sc_2.pkl")

ticker_all = ["2303.TW", "2330.TW", "2317.TW", "2412.TW", "3008.TW"]
pre_all = [0, 0, 0, 0, 0]

# 循环遍历所有股票
i = 0
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = '2024-11-28'

for ticker_ever in ticker_all:
  
    data = yf.download(ticker_ever, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    
    # 使用与训练时一致的特征列
    data_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[data_columns].values.astype('float')
    
    # 归一化数据
    data = sc.transform(data) 
    

    
    # 创建时间步输入数据
    def prepare_prediction_data(data, timestep):
        input_data = []
        for j in range(len(data) - timestep):
            input_data.append(data[j:j + timestep])
        return np.array(input_data)
    
    input_data = prepare_prediction_data(data, timestep)
    input_tensor = torch.from_numpy(input_data).to(torch.float32)
    
    # 模型预测
    with torch.no_grad():
        predictions = model_dcnn(input_tensor.unsqueeze(1))  # 添加通道维度
    
    # 反归一化预测结果
    predictions = predictions.numpy()
    predictions_inverse = sc_2.inverse_transform(predictions)
    
    # 保存当前股票的预测结果
    pre_all[i] = round(predictions_inverse[-1].item(), 2)
    i=i+1


with st.container():
    coldcnn1, coldcnn2,coldcnn3,coldcnn4,coldcnn5 = st.columns(5)
    i=0
    for ticker_ever in ticker_all:
        if i == 0:
            coldcnn1.metric(label=ticker_ever, value=f"{pre_all[i]}")  # 显示结果
        elif i == 1:
            coldcnn2.metric(label=ticker_ever, value=f"{pre_all[i]}")
        elif i == 2:
            coldcnn3.metric(label=ticker_ever, value=f"{pre_all[i]}")
        elif i == 3:
            coldcnn4.metric(label=ticker_ever, value=f"{pre_all[i]}")
        else:
            coldcnn5.metric(label=ticker_ever, value=f"{pre_all[i]}")
        
        i += 1

    




# --------------------------------------------------------------------------------------
st.write("---")
uploaded_file = st.file_uploader("測試")

if uploaded_file is not None:
    ee=[]
    try:
        def transform_data(df):

            data_index =  ['Close','Volume2','h-l','sma','10ema','greed','adr']
        
            flatten_data = df[data_index].values.reshape(-1)  # 攤平資料
            
            # 轉換成字串
            str_data = "<SEP>".join(flatten_data.astype('str'))
            filter_data = str_data.replace(',',"").replace('X',"")
            
            # 切割回陣列
            x_data = filter_data.split("<SEP>")
            
            return x_data
            
        x = []
        df = pd.read_csv(uploaded_file)
        df2=df['Close']
        data = transform_data(df)

        data_index=['Close']
        data2 = df[data_index].values.reshape(-1)  # 攤平資料
        ee.extend(data2)

        x.extend(data)
        sc = MinMaxScaler()
        sc_2 = MinMaxScaler()


        x = np.array(x).astype('float')
        x = x.reshape(-1, len(['Close','Volume2','h-l','sma','10ema','greed','adr']))

        ee = np.array(ee).astype('float')
        ee = ee.reshape(-1, 1)
        ee=sc_2.fit_transform(ee)

        x = sc.fit_transform(x)
        y = x[:,0]


        def split_data(datas, labels, split_num = 10):
            max_len = len(datas)
            x, y = [], []
            for i in range(max_len - split_num -1):
                x.append(datas[i: i+split_num])
                y.append(labels[split_num+i+1])
            
            return np.array(x), np.array(y)


        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=1, shuffle=False)

        x_valid, y_valid = split_data(x_valid, y_valid)

        x_valid=torch.from_numpy(x_valid).type(torch.Tensor)
        y_valid=torch.from_numpy(y_valid).float().unsqueeze(1)
        x_valid = x_valid
        y_valid=y_valid

        with torch.no_grad():
            x_valid = x_valid.to(device)  # Move x_valid to the same device as the model
            y_pred = model_1(x_valid)
        y_pred = y_pred.cpu().numpy()
        y_valid = y_valid.cpu().numpy()
        y_pred_inverse = sc_2.inverse_transform(y_pred)
        y_valid_inverse = sc_2.inverse_transform(y_valid)
        # print(y_pred_inverse.item())#預測
        # print(y_valid_inverse.item())#原始
        

        def display_predictions(y_pred_inverse, y_valid_inverse):
            template = f"""
            ### 預測結果預測:{round(y_pred_inverse.item(),1)}

            

            ### 預測結果原始:{y_valid_inverse.item()}
            
            """
            st.markdown(template)
            st.toast('上傳成功!!', icon='🎉')

        display_predictions(y_pred_inverse, y_valid_inverse)
    except Exception as e:
         st.error(f"文件處理過程中發生錯誤請確認上傳文件格式")
    
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


text_input = st.text_input(
    "Enter some text 👇",
    placeholder="輸入文章",
    label_visibility="visible",
    disabled=False,

)



if text_input:
    from deep_translator import GoogleTranslator
    label_decoding = {0:'negative', 1:'positive'}
   

    user_input = text_input  # 假設 text_input 是用戶輸入的文本
    
    # 使用 deep-translator 進行翻譯，從繁體中文翻譯到英文
    translation = GoogleTranslator(source='zh-TW', target='en').translate(user_input)
    user_input = translation  # 更新為翻譯後的文本

    
    tokenizer = get_tokenizer('basic_english')
    ans=predict_sentiment(user_input)
    if(ans=='positive'):{
        st.balloons()
    }
    else:
        st.snow()
    st.text_area("轉為英文：", user_input, height=200)
    
    st.write("輸入文章情緒：", ans)
    

