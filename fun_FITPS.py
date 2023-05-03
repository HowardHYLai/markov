# FIT-PS python化 完成

# 跟學長matlab的差異
# 判斷[data(i)<0且data(i+1)>=0]或[data(i)>0且data(i+1)<=0]
# 缺值補值(判斷缺半週期或一週期決定補多少)

import matplotlib.pyplot as plt 
import numpy as np 


def FITPS(data,SampingPerCycle):
    # data 訊號源
    #   SampingPerCycle 一週期取樣點數
    # FIT-PS
    Fs=len(data[:,0]) # f signal Sampling rate
    fs =60 # f signal 

    # 檢測電壓過零點
    z=np.ones((len(data[:,0])-1), int)    # 過零點電壓取樣點次序
    zvalue=np.ones((len(data[:,0])-1), float)   # 過零點電壓數值
    zp=0    # 過零點個數-1(-1因為要當索引)    (matalb_z1)
    for vt in np.linspace(0,len(data[:,0])-2,len(data[:,0])-1,dtype=int):
        if (data[vt,0] <= 0 and data[vt+1,0] > 0) or (data[vt,0] >= 0 and data[vt+1,0] < 0) :
            z[zp]=vt*z[zp]
            zvalue[zp]=data[vt,0]*zvalue[zp]
            zp+=1
    z=z[0:zp]    
    zvalue=zvalue[0:zp]

    # 計算各精確過零點所需位移量
    # SampingPerCycle=32
    zshift=np.ones([zp,int(SampingPerCycle/2)], float)   # 各過零點位移量
    # len(z[:,:]) = zp 吧   ? 下面for迴圈
    for zn in np.linspace(0,zp-1,zp,dtype=int):
        z_th=z[zn]
        zshift[zn,0]=-data[z_th,0]/(data[z_th+1,0]-data[z_th,0])
    # print('檢查用:',zshift[:,0])

    # 插值法計算配置週期取樣點
    import math
    SampingPerCycle=32  # SampPerPeriod
    zn=0
    SampingPerCycle_half=SampingPerCycle/2
    z2z=np.ones(len(z)-1, float)  # 零點到下一個零點的距離z2z   (matalb_Len)
    Dis=np.ones(len(z)-1, float)  # 每半週期內兩點之距離
    Voltage=np.ones([len(z)-1,int(SampingPerCycle/2)], float)
    Current=np.ones([len(z)-1,int(SampingPerCycle/2)], float)
    k1=np.ones([len(z)-1,int(SampingPerCycle/2)], float)
    k2=np.ones([len(z)-1,int(SampingPerCycle/2)], float)
    k3=np.ones([len(z)-1,int(SampingPerCycle/2)], float)
    correct_data_len=0
    for zn in np.linspace(0,len(z)-2,len(z)-1,dtype=int):
        # print(zn)
        z2z[zn]=((z[zn+1]+zshift[zn+1,0])-(z[zn]+zshift[zn,0]))*z2z[zn]
        Dis[zn]=z2z[zn]/(SampingPerCycle/2)
        # 插值法校正半週期內所有取樣點值(V,I)
        # zshift_VI=zshift
        for k in np.linspace(0,int(SampingPerCycle/2)-1,int(SampingPerCycle/2),dtype=int):
            k1[zn,k]=z[zn]+zshift[zn,0]+(Dis[zn]*k)
            k2[zn,k]=int(math.floor(k1[zn,k]))
            k3[zn,k]=int(math.ceil(k1[zn,k]))
            if k2[zn,k]==k3[zn,k]:
                k3[zn,k]+=1
            zshift[zn,k]=(k1[zn,k]-k2[zn,k])/(k3[zn,k]-k2[zn,k])
            Voltage[zn,k]=data[int(k2[zn,k]),0]+(data[int(k3[zn,k]),0]-data[int(k2[zn,k]),0])*zshift[zn,k]
            Current[zn,k]=data[int(k2[zn,k]),1]+(data[int(k3[zn,k]),1]-data[int(k2[zn,k]),1])*zshift[zn,k]
            # print(zn,k)
            correct_data_len=(zn+1)*(k+1)


    # 組裝波形
    V_signal=np.ones(correct_data_len, float)
    I_signal=np.ones(correct_data_len, float)
    for kk in np.linspace(0,len(Voltage[:,0])-1,len(Voltage[:,0]),dtype=int):
        V_signal[int(SampingPerCycle/2)*kk:int(SampingPerCycle/2)*(kk+1)]=Voltage[kk,:].reshape(len(Voltage[kk,:]))
        I_signal[int(SampingPerCycle/2)*kk:int(SampingPerCycle/2)*(kk+1)]=Current[kk,:].reshape(len(Current[kk,:]))

    # print('case ?')
    while len(V_signal[:]) != SampingPerCycle*60 :
        # print('case 0')
        if len(V_signal[:]) > SampingPerCycle*60 :  #   超過原長度(1920)
            V_signal=V_signal[0:SampingPerCycle*60]
            I_signal=I_signal[0:SampingPerCycle*60]
            # print('case 1')
        else :  #   <原長度(1920)
            # print('case 2')
            if len(V_signal[:]) % int(SampingPerCycle/2)==0 :   #   缺16的倍數
                if len(V_signal[:]) % SampingPerCycle==0 :  #是32
                    tempV=V_signal[len(V_signal[:])-SampingPerCycle:len(V_signal[:])]
                    tempI=I_signal[len(I_signal[:])-SampingPerCycle:len(I_signal[:])]
                    V_signal=np.append(V_signal,tempV)
                    I_signal=np.append(I_signal,tempI)
                    # print('32')
                else:   # 是16不是32
                    tempV=V_signal[len(V_signal[:])-int(SampingPerCycle/2)*2:len(V_signal[:])-int(SampingPerCycle/2)]
                    tempI=I_signal[len(I_signal[:])-int(SampingPerCycle/2)*2:len(I_signal[:])-int(SampingPerCycle/2)]
                    V_signal=np.append(V_signal,tempV)
                    I_signal=np.append(I_signal,tempI)
                    # print('16')
                    # print('case 3')
            else:   # 不是16    (都不是)
                tempV=V_signal[len(V_signal[:])-int(SampingPerCycle/2):len(V_signal[:])-int(SampingPerCycle/2)+1]
                tempI=I_signal[len(I_signal[:])-int(SampingPerCycle/2):len(I_signal[:])-int(SampingPerCycle/2)+1]
                V_signal=np.append(V_signal,tempV)
                I_signal=np.append(I_signal,tempI)
                # print('case 4')
    V_I_signal=[V_signal,I_signal]
    V_I_signal=np.array(V_I_signal)
    V_I_signal=np.transpose(V_I_signal)
    # print('1')
    return V_I_signal