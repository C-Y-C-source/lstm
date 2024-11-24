
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

from torchtext.data.utils import get_tokenizer
import os
import pandas as pd
import torch
from collections import Counter

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
# if(one_time==0):
#     df = pd.read_csv('lstm_ana\sentimentData\IMDB Dataset.csv')
#     reviews = df['review'].values
#     sentiments = df['sentiment'].values
#     tokenizer = get_tokenizer('basic_english')
#     counter = Counter()
#     for review in reviews:
#         token = tokenizer(review)
#         counter.update(token)
#     one_time=1
#     token_vocab = vocab(counter, min_freq=15, specials=('<pad>', '<unk>'))
#     token_vocab.set_default_index(token_vocab.get_stoi()['<unk>'])

#     PAD_IDX = token_vocab.get_stoi()['<pad>']
#     INPUT_DIM = len(token_vocab)

#     reviews_ids = [torch.tensor(token_vocab.lookup_indices(tokenizer(i))) for i in reviews]
#     labels = (sentiments=='positive').astype('float32')
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

def display_bar_chart():
    # 配置直方圖選項
    option = {
        "backgroundColor": "#212121",
        "title": {
            "text": "各模型模擬交易平均年化率(%)",
            "subtext": "資料範圍:2019/04/19-2024/04/10",
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
            "data": ["年化率"],
            "textStyle": {
                "color": "#f2f2f2"
            }
        },
        "xAxis": {
            "type": "category",
            "data": ["lstm", "cnn", "lstm-GAN", "gru", "gru-GAN","均值回歸(MRS)"],
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
                "name": "年化率",
                "type": "bar",
                "data": [9,15,24,8,16,6],
                "itemStyle": {
                    "color": "#ef4136"
                }
            }
        ]
    }

    # 使用 streamlit-echarts 顯示直方圖
    st_echarts(options=option, height="600px")

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

# 標題
st.title("測試中")

import streamlit as st

# 創建三個列
col1, col2, col3 = st.columns(3)

# 在每個列中放置一個按鈕
with col1:
    clicked1 = st.button("按鈕1", help="選擇模型1")

with col2:
    clicked2 = st.button("按鈕2", help="選擇模型2")

with col3:
    clicked3 = st.button("按鈕3", help="選擇模型3")

# 根據按鈕的點擊狀態執行相應操作
if clicked1:
    st.write("按鈕1被點擊了！")
    #display_bar_chart()

if clicked2:
    st.write("按鈕2被點擊了！")

if clicked3:
    st.write("按鈕3被點擊了！")


#分詞


    





uploaded_file = st.file_uploader("測試")
from google.cloud import translate_v2 as translate
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

import googletrans
if text_input:
    from translate import Translator
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
    user_input=text_input
    translation = translator.translate(user_input, src='zh-tw',dest='en') 
    user_input = translation.text

    
    tokenizer = get_tokenizer('basic_english')
    ans=predict_sentiment(user_input)
    if(ans=='positive'):{
        st.balloons()
    }
    else:
        st.snow()
    
    st.write("輸入文章情緒：", ans)

