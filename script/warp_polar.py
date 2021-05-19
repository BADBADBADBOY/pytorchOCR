#-*- coding:utf-8 _*-
"""
@author:fxw
@file: tt.py
@time: 2020/12/25
"""
import cv2
import numpy as np
import sys

#实现图像的极坐标的转换 center代表及坐标变换中心‘;r是一个二元元组，代表最大与最小的距离；theta代表角度范围
#rstep代表步长； thetastap代表角度的变化步长
def polar(image,center,r,theta=(70,360+70),rstep=0.8,thetastep=360.0/(360*2)):
    #得到距离的最小值、最大值
    minr,maxr=r
    #角度的最小范围
    mintheta,maxtheta=theta
    #输出图像的高、宽 O:指定形状类型的数组float64
    H=int((maxr-minr)/rstep)+1
    W=int((maxtheta-mintheta)/thetastep)+1
    O=125*np.ones((H,W,3),image.dtype)
    #极坐标转换  利用tile函数实现W*1铺成的r个矩阵 并对生成的矩阵进行转置
    r=np.linspace(minr,maxr,H)
    r=np.tile(r,(W,1))
    r=np.transpose(r)
    theta=np.linspace(mintheta,maxtheta,W)
    theta=np.tile(theta,(H,1))
    x,y=cv2.polarToCart(r,theta,angleInDegrees=True)
    #最近插值法
    for i in range(H):
        for j in range(W):
            px=int(round(x[i][j])+cx)
            py=int(round(y[i][j])+cy)
            if((px>=0 and px<=w-1) and (py>=0 and py<=h-1)):
                O[i][j][0]=image[py][px][0]
                O[i][j][1]=image[py][px][1]
                O[i][j][2]=image[py][px][2]

    return O

import time
if __name__=="__main__":
    img = cv2.imread(r"C:\Users\fangxuwei\Desktop\111.jpg")
    # 传入的图像宽：600  高：400
    h, w = img.shape[:2]
    print("h:%s w:%s"%(h,w))
    # 极坐标的变换中心（300，200）
    # cx, cy = h//2, w//2
    cx, cy = 204, 201
    # cx, cy = 131, 123
    # 圆的半径为10 颜色：灰 最小位数3
    cv2.circle(img, (int(cx), int(cy)), 10, (255, 0, 0, 0), 3)
    s = time.time()
    L = polar(img, (cx, cy), (h//3, w//2))
    # 旋转
    L = cv2.flip(L, 0)
    print(time.time()-s)

    # 显示与输出
    cv2.imshow('img', img)
    cv2.imshow('O', L)
    cv2.waitKey(0)





