from pyts.datasets import load_gunpoint
from pyts.image import MarkovTransitionField
from pyts.image import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import copy
from fun_FITPS import FITPS

V, I = None, None

path = r"C:\NILM\DATA\ya.csv"
data = pd.read_csv(path)
V = list(data.iloc[:,0]) #實驗室量測數據用
I = list(data.iloc[:,1])

# V = list(data.iloc[:,1]) #實驗室量測數據用
# I = list(data.iloc[:,0])


## 電壓過零點 
zeros = []

# size = len(V)
# for i in range(size-1):
#     if V[i] < 0 and V[i+1]>=0:
#         zeros.append(i+1)

for i in range (0,len(V)-1):                        #計算過零點
    if V[i+1]>0 and V[i]<0:                         #判斷式1
        zeros.append(i)                              #將值丟入矩陣
    elif V[i+1]>0 and V[i-1]<0 and V[i] == 0:       #判斷式2
        zeros.append(i)

## fitps 
##########################################################################
V_fitps = np.array([])
I_fitps = np.array([])

size = len(zeros)
## 以一秒為單位做FITPS轉換
for i in range(0, size-60, 60):
    start, end = zeros[i], zeros[i+60]
    waves = np.array([V[start:end], I[start:end]]).T 
    waves_fitps = FITPS(waves, 32)  ##32 
    V_fitps = np.append(V_fitps, waves_fitps[:,0])
    I_fitps = np.append(I_fitps, waves_fitps[:,1])

print("V_fitps長度 : ", len(V_fitps))
print("I_fitps長度 : ", len(I_fitps))

plt.plot(I[1680:1703])
plt.title("I_ORIGINAL")
plt.show()


plt.plot(I_fitps)
plt.title("I_FITPS")
plt.show()


plt.plot(V_fitps[:33])
plt.title("T")
plt.show()
##########################################################################

## 處理資料
x = []
size = len(V_fitps)
for i in range(0, size, 32):
    start, end = i, i+32
    
    x.append(I_fitps[start:end])


## Markov
X = np.array(x) ## shape: (6817, 32)  
print("X的維度 : ",X.shape)  

# mtf = MarkovTransitionField(n_bins=8)     #Markov
# y = mtf.fit_transform(x)

# mtf = MarkovTransitionField(image_size=32,n_bins=8)
# y = mtf.fit_transform(x)

rp = RecurrencePlot(dimension=3, time_delay=1)   #RecurrencePlot
y = rp.fit_transform(x)

# rp = RecurrencePlot(threshold='point', percentage=30)
# y = rp.fit_transform(x)


# 畫圖
# plt.imshow(y[175])  ## 
# plt.show()

# for i in range (183, 391):
#     plt.imshow(y[i])

#     # plt.savefig("C:/NILM/pictur/" )  #儲存圖片

#     plt.savefig(r"C:\NILM\pictur_for_code\612_ele\marcov\吸塵+冷氣\測試/{}.png".format(i)) #输入地址，并利用format函数修改图片名称
#     plt.clf() #需要重新更新画布，否则会出现同一张画布上绘制多张图片

# #     # plt.show()