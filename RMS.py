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


# ======================================
# 移動視窗
# ======================================

# x = []
# size = len(rms)
# for i in range(0, size, 32):
#     start, end = i, i+32
    
#     x.append([rms[start:end]])

x = []  #1077
size = len(rms)

for i in range(0, size - 32, 10):
    a = (rms[i : i + 32])
    # b = abs(a)
    x.append(a)



X = np.array(x) ## shape: (6817, 32)  
print("X的維度 : ",X.shape)
  

# mtf = MarkovTransitionField(n_bins=8)     #Markov
# y = mtf.fit_transform(x)

mtf = MarkovTransitionField(n_bins=8)
y = mtf.fit_transform(x)
# y = mtf.transform(x)

# rp = RecurrencePlot(dimension=3, time_delay=1)   #RecurrencePlot
# y = rp.fit_transform(x)

# rp = RecurrencePlot(threshold='point', percentage=30)
# y = rp.fit_transform(x)


# 畫圖
# plt.imshow(y[1])  ## 
# plt.show()

# for i in range (0, len(y)):
#     plt.imshow(y[i])

#     # plt.savefig("C:/NILM/pictur/" )  #儲存圖片

#     plt.savefig(r"C:\NILM\pictur_for_code\cool數據集\RMS\覆蓋率 32-10\cool hair dry/{}.png".format(i)) #输入地址，并利用format函数修改图片名称
#     plt.clf() #需要重新更新画布，否则会出现同一张画布上绘制多张图片

# #     # plt.show()