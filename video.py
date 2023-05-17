import numpy as np
import cv2
size = (640,480)#这个是图片的尺寸，一定要和要用的图片size一致
#完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
videowrite = cv2.VideoWriter(r"C:\NILM\pictur_for_code\612_ele\video\test",-1,20,size)#20是帧数，size是图片尺寸
img_array=[]
for filename in (r"C:\NILM\pictur_for_code\612_ele\marcov\吸塵+冷氣\開吸塵1\{}.png".format(i) for i in range(68,107)):#这个循环是为了读取所有要用的图片文件
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
for i in range(39):#把读取的图片文件写进去
    videowrite.write(img_array[i])
videowrite.release()
print('end!')
