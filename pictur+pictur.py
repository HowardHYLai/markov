import cv2
# img = cv2.imread(r"C:\NILM\pictur_for_code\612_ele\re\hair_dry\4\open\4831.png")
# logo = cv2.imread(r"C:\NILM\pictur_for_code\612_ele\marcov\hair_dry\open\4831.png")
# output = cv2.addWeighted(img, 0.5, logo, 0.3, 50)

# cv2.imshow('oxxostudio', output)
# cv2.waitKey(0)      # 按下任意鍵停止
# cv2.destroyAllWindows()

img_red = cv2.imread(r"C:\NILM\pictur_for_code\612_ele\re\hair_dry\4\open\4832.png")
img_green = cv2.imread(r"C:\NILM\pictur_for_code\612_ele\marcov\hair_dry\4\open\4832.png")
# img_blue = cv2.imread('test-blue.png')

output = cv2.add(img_red, img_green)  # 疊加紅色和綠色
# output = cv2.add(output, img_blue)    # 疊加藍色

cv2.imshow('oxxostudio', output)
cv2.waitKey(0)     # 按下任意鍵停止
cv2.destroyAllWindows()

Img_Name = (r"C:\NILM\pictur_for_code\612_ele\pictur+pictur\hair_dry(markov+re)/{}.png".format(4832))
cv2.imwrite(Img_Name  , output)