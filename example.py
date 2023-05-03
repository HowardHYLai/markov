# -*- coding: utf-8 -*-



from pyts.datasets import load_gunpoint
from pyts.image import MarkovTransitionField
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import copy

V, I = None, None

path = r"./0001Hairdry_vi.csv"
data = pd.read_csv(path)
V = list(data.iloc[:,0])  ## 電壓
I = list(data.iloc[:,1])  ## 電流


x = []

for i in range(0, len(I)-32, 32):  
    start, end = i, i+32   ## 照理說要先取 電壓過零點->fitps降採樣到32
    x.append(I[start:end])


X = np.array(x) ## shape: (6817, 32)  第二維不能有 32、33這樣跳，要統一大小

print("X的維度 : ",X.shape)  

mtf = MarkovTransitionField(n_bins=8)
y = mtf.fit_transform(x)



plt.imshow(y[10])  ## 畫第10個週期 

