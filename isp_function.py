# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:04:34 2021

@author: 碉堡了的少年
"""

# isp函数

import numpy as np
from scipy.interpolate import interp1d
import cv2

M_rgb2yuv=np.array([[0.299,0.587,0.114],
                    [-0.169,-0.331,0.499],
                    [0.499,-0.418,-0.081]])

M_yuv2rgb=np.array([[9.99999554e-01, -4.46062343e-04,1.40465882],
                     [9.99655449e-01, -3.44551299e-01,-7.15683665e-01],
                     [1.00177531e+00,1.77530689,9.94081794e-04]])

M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                    [-0.96924364,1.8759675,0.04155506],
                    [0.05563008,-0.20397695,1.05697151]])
lab_ideal=np.array( # x-rite 色彩标准，X-Rite官网提供的LAB色彩真值
    [[37.986,13.555,14.059],#1,1 dark skin 深肤色
      [65.711,18.13,17.81],#1,2 light skin 浅肤色
      [49.927,-4.88,-21.925],#1,3 blue sky蓝天
      [43.139,-13.095,21.905],#1,4 foliage 绿叶
      [55.112,8.844,-25.399],#1,5 blue flower 蓝花
      [70.719,-33.397,-0.199],#1,6 bluish green 蓝绿色
      [62.661,36.067,57.096],#2,1 orange 橙色
      [40.02,10.41,-45.964],#2,2 purplish blue 紫蓝色
      [51.124,48.239,16.248],#2,3 moderate red 粉红色
      [30.325,22.976,-21.587],#2,4 purple 紫色
      [72.532,-23.709,57.255],#2,5 yellow green 黄绿色
      [71.941,19.363,67.857],# 2,6 orange yellow 橙黄色
      [28.778,14.179,-50.297],#3,1 bluw 蓝色
      [55.261,-38.342,31.37],#3,2 green 绿色
      [42.101,53.378,28.19],#3,3 red 红色
      [81.733,4.039,79.819],#3,4 yellow 黄色
      [51.935,49.986,-14.574],#3,5 magenta 品红
      [51.038,-28.631,-28.638],#3,6 cyan 青色
      [96.539,-0.425,1.186],
      [81.257,-0.638,-0.335],
      [66.766,-0.734,-0.504],
      [50.867,-0.153,-0.27],
      [35.656,-0.421,-1.231],
      [20.461,-0.079,-0.973]],dtype='float32')

def rawread(file_path, size=(4208,3120),bayer='BG',OB=1024):
    img=np.fromfile(file_path,dtype='uint16')
    img=img.reshape((size[1], size[0]))
    if   bayer=='RG':
        bayer_form=cv2.COLOR_BAYER_RG2RGB
    elif bayer=='BG': 
        bayer_form=cv2.COLOR_BAYER_BG2RGB
    elif bayer=='GR':
        bayer_form=cv2.COLOR_BAYER_GR2RGB
    elif bayer=='GB':
        bayer_form=cv2.COLOR_BAYER_GB2RGB
    img=cv2.cvtColor(img,bayer_form)
    img=img.astype(np.float32)
    img=(img-OB)/(65535-OB)
    img[img<0]=0
    return img
 
def gamma(x,colorspace='sRGB'):
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgb'):
        y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
        y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    elif colorspace in ('TP', 'my'):  
        y[ (x>=0)&(x<=1)]=(1.42*(1-(0.42/(x[(x>=0)&(x<=1)]+0.42))))
    elif (type(colorspace)==float)|(type(colorspace)==int):
        beta=colorspace
        y[ (x>=0)&(x<=1)]=((1+beta)*(1-(beta/(x[(x>=0)&(x<=1)]+beta))))
    return y

def gamma_reverse(x,colorspace= 'sRGB'):
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
        y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
        y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4
    elif colorspace in ('TP','my'):
        y[(x>=0)&(x<=1)]=0.42/(1-(x[(x>=0)&(x<=1)]/1.42))-0.42         
    return y

def im2vector(img):
    size=img.shape
    rgb=np.reshape(img,(size[0]*size[1],3))
    func_reverse=lambda rgb : np.reshape(rgb,(size[0],size[1],size[2]))
    return rgb, func_reverse    

def awb(img, awb_para):  
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)
    if type(awb_para)==list:
        awb_para=np.array(awb_para)
    if (awb_para>=20).all():
        awb_para=awb_para/1024
    rgb[:,0]=rgb[:,0]*awb_para[0]    
    rgb[:,1]=rgb[:,1]*awb_para[1]    
    rgb[:,2]=rgb[:,2]*awb_para[2]    
    img=func_reverse(rgb)    
    return img

def ccm(img, ccm):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)
    if type(ccm)==list:
        ccm=np.array(ccm)
    if (ccm>=1024).any():
        ccm=ccm/1024
    rgb=rgb.transpose()
    rgb=ccm@rgb
    rgb=rgb.transpose()    
    img_out=func_reverse(rgb)    
    return img_out

def rgb2yuv(img):
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img, img, img))    
            (rgb,func_reverse)=im2vector(img)
    rgb=rgb.transpose()    
    yuv=M_rgb2yuv@rgb
    yuv=yuv.transpose()
    img_out=func_reverse(yuv)
    return img_out

def yuv2rgb(img):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x   
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)    
    rgb=rgb.transpose()    
    rgb=M_yuv2rgb@rgb    
    rgb=rgb.transpose()    
    img_out=func_reverse(rgb)    
    return img_out

def rgb2hue(rgb):
    r=rgb[:,0]
    g=rgb[:,1]
    b=rgb[:,2]
    theta=np.zeros(r.shape)
    flag=~((r==g)&(g==b)&(r==b)) #BUG 在下方的test中，有几个像素转换出现了NAN
    theta[flag]=np.arccos((1/2*((r[flag]-g[flag])+(r[flag]-b[flag])))/(((r[flag])**2+(g[flag])**2+(b[flag])**2-r[flag]*g[flag]-b[flag]*g[flag]-r[flag]*b[flag])**(1/2)))
    theta=theta/np.pi*180
    hue=np.zeros(theta.shape)
    hue[b<=g]=theta[b<=g]    
    hue[b>g]=360-theta[b>g]
    return hue 

def rgb2hsv(img):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img.copy()
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb, func_reverse)=im2vector(img)    
    r=rgb[:,0]
    g=rgb[:,1]
    b=rgb[:,2]
    theta=np.zeros(r.shape)
    # flag=~((r==g)&(g==b)&(r==b)) #BUG 在下方的test中，有几个像素转换出现了NAN
    flag=~(g==b) # g=b的情况下arccos变量可能因为浮点数计算误差导致<-1或者>1,导致arccos无定义
    theta[flag]=np.arccos((1/2*((r[flag]-g[flag])+(r[flag]-b[flag])))/(((r[flag])**2+(g[flag])**2+(b[flag])**2-r[flag]*g[flag]-b[flag]*g[flag]-r[flag]*b[flag])**(1/2)))
    theta=theta/np.pi*180
    flag=(r>=g)&(g==b)
    theta[flag]=0
    flag=(r<g)&(g==b)
    theta[flag]=180
    h=np.zeros(theta.shape)
    h[b<=g]=theta[b<=g]    
    h[b>g]=360-theta[b>g]
    s=np.zeros(theta.shape)
    max_rgb=np.maximum(r,np.maximum(g,b))
    min_rgb=np.minimum(r,np.minimum(g,b))
    s=(max_rgb-min_rgb)/max_rgb
    v=max_rgb
    hsv=np.zeros(rgb.shape)
    hsv[:,0],hsv[:,1],hsv[:,2]=h,s,v    
    img_out=func_reverse(hsv)        
    return img_out 

def hsv2rgb(img):
    if (img.shape[1]==3)&(img.ndim==2):
        hsv=img.copy()
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (hsv, func_reverse)=im2vector(img)    
    h,s,v=hsv[:,0],hsv[:,1],hsv[:,2]
    h_i=np.floor(h/60)
    f=h/60-h_i
    p=v*(1-s)
    q=v*(1-f*s)
    t=v*(1-(1-f)*s)
    r=np.zeros(hsv.shape[0])
    g=np.zeros(hsv.shape[0])
    b=np.zeros(hsv.shape[0])
    idx=(h_i==0)
    r[idx],g[idx],b[idx]=v[idx],t[idx],p[idx]
    idx=(h_i==1)
    r[idx],g[idx],b[idx]=q[idx],v[idx],p[idx]
    idx=(h_i==2)
    r[idx],g[idx],b[idx]=p[idx],v[idx],t[idx]
    idx=(h_i==3)
    r[idx],g[idx],b[idx]=p[idx],q[idx],v[idx]
    idx=(h_i==4)
    r[idx],g[idx],b[idx]=t[idx],p[idx],v[idx]
    idx=(h_i==5)
    r[idx],g[idx],b[idx]=v[idx],p[idx],q[idx]
    img_out=func_reverse(np.vstack((r,g,b)).transpose())
    return img_out 

def hsv(img,hue_lut=0,sat_lut=64):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img.copy()
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb, func_reverse)=im2vector(img)
    hue=rgb2hue(rgb)    
    func_theta=interp1d(np.linspace(0,360,len(hue_lut)+1),np.hstack((hue_lut,hue_lut[0]))/64*60/180*np.pi)
    func_alpha=interp1d(np.linspace(0,360,len(sat_lut)+1),np.hstack((sat_lut,sat_lut[0]))/64)
    theta=func_theta(hue)    
    alpha=func_alpha(hue)    
    for idx in range(rgb. shape[0]):
        tmp_rgb=rgb[idx,:]
        tmp_theta=theta[idx]
        tmp_alpha=alpha[idx]
        tmp_yuv=M_rgb2yuv@tmp_rgb
        tmp_M=np.array([[1,0,0],      
                        [0,tmp_alpha*np.cos(tmp_theta),-tmp_alpha*np.sin(tmp_theta)],      
                        [0,tmp_alpha*np.sin(tmp_theta) ,tmp_alpha*np.cos (tmp_theta)]])
        tmp_yuv=tmp_M@tmp_yuv      
        tmp_rgb=M_yuv2rgb@tmp_yuv
        rgb[idx,:]=tmp_rgb
    img_out=func_reverse(rgb)        
    return img_out

def gaussian(R, sigma=1):
    if sigma!=0:
        f=lambda x : 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x**2)/2/(sigma**2))
        R=int(np.ceil(R))
        (x,y)=np.meshgrid(np.arange(-R,R+1),np.arange(-R,R+1))
        d=np.hypot(x,y)
        w=f(d)
        w=w/w.sum()
    else:
        w=np.ones((1,1))
    return w

def xyz2lab(img,whitepoint='D65'):
    if (img.ndim==3):
        if (img.shape[2]==3):
            (xyz,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            xyz=img
            func_reverse=lambda x : x
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
        (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    L=116*f(xyz[:,1]/Yn)-16
    a=500*(f(xyz[:,0]/Xn)-f(xyz[:,1]/Yn))
    b=200*(f(xyz[:,1]/Yn)-f(xyz[:,2]/Zn))
    Lab=np.vstack((L,a,b)).transpose()
    img_out=func_reverse(Lab)
    return img_out

def rgb2lab(img,whitepoint='D65'):
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb,Func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
        (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    L=116*f(xyz[:,1]/Yn)-16
    a=500*(f(xyz[:,0]/Xn)-f(xyz[:,1]/Yn))
    b=200*(f(xyz[:,1]/Yn)-f(xyz[:,2]/Zn))
    Lab=np.vstack((L,a,b)).transpose()
    img_out=func_reverse(Lab)
    return img_out

def lab2xyz(img,whitepoint='D65'):
    if (img.ndim==3):
        if (img.shape[2]==3):
            (lab,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            lab=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (lab,Func_reverse)=im2vector(img)
    lab=lab.transpose()
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    f_reverse=lambda t : (t>(6/29))*(t**3)+\
        (t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    xyz=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200) ))
    xyz=xyz.transpose()
    img_out=func_reverse(xyz)
    return img_out

def lab2rgb(img,whitepoint='D65'):
    if (img.ndim==3):
        if (img.shape[2]==3):
            (lab,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            lab=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (lab,Func_reverse)=im2vector(img)
    lab=lab.transpose()
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    f_reverse=lambda t : (t>(6/29))*(t**3)+\
        (t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    xyz=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200) ))
    rgb=M_xyz2rgb@xyz
    rgb=rgb.transpose()
    rgb=gamma(rgb,colorspace='sRGB')
    img_out=func_reverse(rgb)
    return img_out

def lab_background():
    import matplotlib.pyplot as plt
    a,b=np.meshgrid(np.arange(-60,80,0.25),np.arange(100,-60,-0.25))
    L=np.ones(a.shape)*80
    img=lab2rgb(np.dstack((L,a,b)))
    plt.figure(tight_layout=True,figsize=(7,8))
    plt.axes(xlim=[-60,80],ylim=[-60,100])
    plt.grid('on')
    plt.imshow(img,extent=(-60,80,-60,100))
    return

def impoly(img,poly_position=None): #四边形框选图像ROI
    "(rgb_mean,rgb_std,poly_position)=impoly(img)\n(rgb_mean,rgb_std,poly_position)=impoly(img,poly_position)"
    import matplotlib.pyplot as plt
    if poly_position is None:
        fig=plt.figure(figsize=[12.,7.5],tight_layout=True)
        plt.imshow(img)
        fig.show()
        # fig.canvas.set_window_title('waiting. ..')
        fig.canvas.manager.set_window_title('waiting. ..')
        pos=plt.ginput(n=4)
        # plt.close(fig)
    else:
        pos=poly_position
    (n,m)=np.meshgrid(np.arange(0.5,6.5)/6,np.arange(0.5,4.5)/4)
    n=n.flatten()
    m=m.flatten()
    x_center=(1-m)*((1-n)*pos[0][0]+n*pos[1][0])+m*(n*pos[2][0]+(1-n)*pos[3][0])
    y_center=(1-m)*((1-n)*pos[0][1]+n*pos[1][1])+m*(n*pos[2][1]+(1-n)*pos[3][1])
    r_sample=np.floor(min([abs(pos[1][0]-pos[0][0])/6,
                           abs(pos[2][0]-pos[3][0])/6,
                           abs(pos[1][1]-pos[2][1])/4,
                           abs(pos[0][1]-pos[3][1])/4]))*0.2
    if poly_position is None:
        plt.plot(pos[0][0],pos[0][1],'r+')
        plt.plot(pos[1][0],pos[1][1],'r+')
        plt.plot(pos[2][0],pos[2][1],'r+')
        plt.plot(pos[3][0],pos[3][1],'r+')
        # plt.plot(x_center,y_center,'yo')
        plt.plot(x_center-r_sample,y_center-r_sample,'y+')
        plt.plot(x_center+r_sample,y_center-r_sample,'y+')
        plt.plot(x_center-r_sample,y_center+r_sample,'y+')
        plt.plot(x_center+r_sample,y_center+r_sample,'y+')
        fig.show()
        poly_position=pos
    else:
        pass
    rgb_mean=np.zeros((24,3))   
    rgb_std=np.zeros((24,3))   
    for block_idx in range(24):
        block=img[np.int(y_center[block_idx]-r_sample):np.int(y_center[block_idx]+r_sample),
                  np.int(x_center[block_idx]-r_sample):np.int(x_center[block_idx]+r_sample),:]
        rgb_vector,_=im2vector(block)
        rgb_mean[block_idx,:]=rgb_vector.mean(axis=0)
        rgb_std[block_idx,:]=rgb_vector.std(axis=0)
    return (rgb_mean,rgb_std,poly_position)
    
#%%
#%% test
# import matplotlib.pyplot as plt
# file_path=r"\\192.168.0.103\Extra\working\0624_d&n\A_day_image[US=5482,AG=1024,DG=1026,R=1,G=0,B=1024]_1920x1080_16_RG_0624144422.jpg"
# img=plt.imread(file_path)[0::1,0::1,:].astype('float32')/255
# plt.figure()
# plt.imshow(img,vmin=0,vmax=1)
# img1=hsv(img,hue_lut=np.array([0]),sat_lut=np.array([64]))
# plt.figure()
# plt.imshow(img1,vmin=0,vmax=1)
#%% test
# import matplotlib.pyplot as plt
# file_path=r"\\192.168.0.103\Extra\working\0624_d&n\A_day_image[US=5482,AG=1024,DG=1026,R=1,G=0,B=1024]_1920x1080_16_RG_0624144422.jpg"
# img=plt.imread(file_path).astype('float32')/255
# rgb_mean,rgb_std,_=impoly(img)
# print(rgb_mean)
# print(rgb_std)
#%%
# plt.figure()
# plt.imshow(gaussian(101,50),cmap='gray')
#%%
# import matplotlib.pyplot as plt
# file_path=r"\\192.168.0.103\Extra\working\0624_d&n\A_day_image[US=5482,AG=1024,DG=1026,R=1,G=0,B=1024]_1920x1080_16_RG_0624144422.jpg"
# img=plt.imread(file_path).astype('float32')/255
# plt.figure()
# plt.imshow(img,vmin=0,vmax=1)
# lab=rgb2lab(img)
# plt.figure()
# plt.imshow(lab[:,:,0],cmap='gray')
#%%
# rgb=np.array([[0,1,0]])
# print(rgb2lab(rgb))
#%%
# import matplotlib.pyplot as plt
# img=plt.imread(r"C:\Users\Administrator\Pictures\IMG_1432.JPG").astype('float32')/255
# lab=rgb2lab(img)
# img_re=lab2rgb(lab)




