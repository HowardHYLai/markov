import pandas as pd
import numpy as np
import math
from pyts.image import MarkovTransitionField
from pyts.image import RecurrencePlot
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt



path = r"C:\NILM\dataset\cooldata\transfer\Hair_drayer_1_0ms.csv"#資料路徑
zero = []                                           #過零點值存放矩陣

data = pd.read_csv(path)                            #pd抓取資料
voltage = data.iloc[:,:1]                           #抓取資料中第一行
V = np.array(voltage[:])                            #轉換成陣列

current = data.iloc[:,1:2]
I = np.array(current[:])

for i in range (0,len(V)-1):                        #計算過零點
    if V[i+1]>0 and V[i]<0:                         #判斷式1
        zero.append(i)                              #將值丟入矩陣
    elif V[i+1]>0 and V[i-1]<0 and V[i] == 0:       #判斷式2
        zero.append(i)

print(len(zero))

# path = r"C:\NILM\DATA\ya.csv"             #資料路徑
# zero = []                                           #過零點值存放矩陣

# data = pd.read_csv(path)                            #pd抓取資料
# voltage = data.iloc[:,:1]                           #抓取資料中第一行
# V = np.array(voltage[:])                            #轉換成陣列

# current = data.iloc[:,1:2]
# I = np.array(current[:])

# for i in range (0,len(V)-1):                        #計算過零點
#     if V[i+1]>0 and V[i]<0:                         #判斷式1
#         zero.append(i)                              #將值丟入矩陣
#     elif V[i+1]>0 and V[i-1]<0 and V[i] == 0:       #判斷式2
#         zero.append(i)




rms = []
for i in range(len(zero)-1):
    start, end = zero[i], zero[i+1]
    rms.append(math.sqrt(sum(x ** 2 for x in I[start:end]) / (end-start)))

# print(rms)

plt.plot(rms)

plt.show()

############################################################

def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size:
            smoothed_data.append(sum(data[0:i+1]) / (i + 1))
        else:
            smoothed_data.append(sum(data[i-window_size+1:i+1]) / window_size)
    return smoothed_data

if __name__ == "__main__":
    # 測試資料，可以自行替換為你的一維時序資料
    input_data = rms

    # 設定移動平均窗口大小，這個數字可以調整來控制平滑程度
    window_size = 5

    # 呼叫移動平均函式進行平滑化
    smoothed_data = moving_average(input_data, window_size)

    # 輸出平滑化後的結果
    print("原始資料：", input_data)
    print("平滑化後的資料：", smoothed_data)

plt.plot(smoothed_data)
plt.title('smoothed_data')
plt.show()


# def exponential_moving_average(data, alpha):
#     smoothed_data = [data[0]]  # 初始點保持不變
#     for i in range(1, len(data)):
#         smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[-1])
#     return smoothed_data

# def main():
#     # 輸入一維時序數據
#     data = rms

#     # 設定平滑參數，alpha是指數移動平均法的權重
#     alpha = 0.1

#     # 使用指數移動平均法進行平滑化
#     smoothed_data = exponential_moving_average(data, alpha)

#     # 輸出平滑後的數據
#     print("原始數據：", data)
#     print("平滑後的數據：", smoothed_data)

#     plt.plot(smoothed_data)
#     plt.title('smoothed_data')
#     plt.show()

# if __name__ == "__main__":
#     main()









############################################################


# ======================================
# 移動視窗
# ======================================

# x = []
# size = len(rms)
# for i in range(0, size, 32):
#     start, end = i, i+32
    
#     x.append([rms[start:end]])
######################################################


x = []  #1077
size = len(smoothed_data)

for i in range(0, size - 20, 10):
    a = (smoothed_data[i : i + 20])
    # b = abs(a)
    x.append(a)



# x = []  #1077
# size = len(rms)

# for i in range(0, size - 32, 4):
#     a = (rms[i : i + 32])
#     # b = abs(a)
#     x.append(a)



X = np.array(x) ## shape: (6817, 32)  
print("X的維度 : ",X.shape)
  


mtf = MarkovTransitionField(n_bins=5)
y = mtf.fit_transform(x)



# 畫圖
# plt.imshow(y[1])  ## 
# plt.show()

for i in range (0, len(y)):
    plt.imshow(y[i])

    # plt.savefig("C:/NILM/pictur/" )  #儲存圖片

    plt.savefig(r"C:\Users\USER\OneDrive\桌面\RMS 不同視窗大小 覆蓋率50%\平滑後\20/{}.png".format(i)) #输入地址，并利用format函数修改图片名称
    plt.clf() #需要重新更新画布，否则会出现同一张画布上绘制多张图片
