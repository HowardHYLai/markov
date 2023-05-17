import numpy as np
# ==============================================================

# def mtf(input_matrix, n_bins):
#     # 对输入矩阵进行线性规划
#     min_val = np.min(input_matrix)
#     max_val = np.max(input_matrix)
#     input_matrix = (input_matrix - min_val) / (max_val - min_val)

#     # 将输入矩阵按 n_bins 分组
#     bins = np.linspace(0, 1, n_bins+1)
#     digitized = np.digitize(input_matrix, bins)

#     # 计算转移矩阵
#     mtf_matrix = np.zeros((n_bins, n_bins))
#     for i in range(digitized.shape[0]-1):
#         row_idx = digitized[i] - 1
#         col_idx = digitized[i+1] - 1
#         mtf_matrix[row_idx, col_idx] += 1

#     # 将转移矩阵标准化
#     row_sums = mtf_matrix.sum(axis=1, keepdims=True)
#     mtf_matrix = mtf_matrix / row_sums

#     return mtf_matrix

# import numpy as np
# import matplotlib.pyplot as plt

# # 定义正弦波的参数
# sampling_rate = 1000
# duration = 10  # 单位：秒
# freq = 1  # 正弦波的频率

# # 生成正弦波信号
# time = np.arange(0, duration, 1/sampling_rate)
# amplitude = np.sin(2 * np.pi * freq * time)

# # 计算正弦波信号的 MTF 矩阵
# mtf_matrix = mtf(amplitude.reshape(-1, 1), n_bins=10)

# # 绘制 MTF 矩阵的热图
# plt.imshow(mtf_matrix, cmap='coolwarm')
# plt.colorbar()
# plt.title('Markov Transition Field of a Sinusoidal Signal')
# plt.xlabel('Current Bin')
# plt.ylabel('Next Bin')
# plt.show()

# =====================================================================











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
    waves_fitps = FITPS(waves, 32)  ## 
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

def calculate_transition_matrix(data, num_bins):
    # 计算转移矩阵
    transition_matrix = np.zeros((num_bins, num_bins))
    for i in range(len(data)-1):
        current_state = int(data[i] * num_bins)
        next_state = int(data[i+1] * num_bins)
        transition_matrix[current_state, next_state] += 1
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True) # 归一化
    return transition_matrix

def generate_mtf_image(data, num_bins, image_size):
    # 生成MTF图像
    transition_matrix = calculate_transition_matrix(data, num_bins)
    mtf_image = np.zeros((image_size, image_size))
    step_size = 1.0 / image_size
    for i in range(image_size):
        for j in range(image_size):
            x = i * step_size
            y = j * step_size
            x_state = int(x * num_bins)
            y_state = int(y * num_bins)
            mtf_image[i, j] = transition_matrix[x_state, y_state]
    return mtf_image

# # 读取CSV文件
# csv_file = "C:/NILM/DATA/ya.csv"
# df = pd.read_csv(csv_file)
# data1 = df.iloc[:, 0].values  # 读取第一列数据
# data2 = df.iloc[:, 1].values  # 读取第二列数据

# MTF 参数
num_bins = 10  # 转移矩阵的大小，即状态的数量
image_size = 100  # MTF 图像的大小

# 生成MTF图像
mtf_image = generate_mtf_image(data, num_bins, image_size)

# 显示图像
plt.imshow(mtf_image, cmap='gray')
plt.colorbar()
plt.show()


