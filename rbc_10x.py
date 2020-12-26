from imutils import paths
import argparse
import cv2
import csv
import os
import numpy as np
# import pandas as pd  
import matplotlib.pyplot as plt
import imutils

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
low=0
up=5
count=0
redious_total=0
r_total=[]
for file in os.listdir('C:/Users/Asus/Desktop/data_3.10/10x/341_2'): 
  max_shar=0 
  Path='C:/Users/Asus/Desktop/data_3.10/10x/341_2/'+file
  print(file)  
  for img in os.listdir(Path):
    imagePath=Path+"/"+img
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    shar=variance_of_laplacian(cl1)
    if shar>max_shar:
      max_shar=shar
      file1=img
  print(file1)  
  imagePath1=Path+"/"+file1 
  zeros=np.zeros((450,700,3))
  img = cv2.imread(imagePath1)
  # img=img[120:360,160:480,:]
  img=img[150:600,200:800,:]
  pic1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  pic2=cv2.Laplacian(pic1, cv2.CV_16S, ksize=1)
  pic2=pic2 - np.min(pic2)
  pic2 = (pic2 / np.max(pic2)) * 255
  pic2 = np.uint8(pic2)
  m = np.copy(img)
  clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(12,12))
  cl1 = clahe.apply(pic1)
  thresh1 = cv2.adaptiveThreshold(cl1 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,1)
  kernel = np.ones((1,1),np.uint8)
  kernel2 = np.ones((2,2),np.uint8)
  kernel1 = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel2, iterations =1)
  closing=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel1, iterations =1)
  sure_bg = cv2.dilate(thresh1,kernel,iterations=3)
  sure_bg2 = cv2.dilate(thresh1,kernel2,iterations=1)
  sure_bg3 = cv2.erode(thresh1,kernel2,iterations=1)
  dist_transform = cv2.distanceTransform(thresh1,cv2.DIST_L1,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg2,sure_fg)
  ret, markers = cv2.connectedComponents(sure_fg)
  markers = markers+10
  markers[unknown==255] = 0
  markers = cv2.watershed(img,markers)
  img[markers == -1] = [255,0,0]
  labels = markers

  i = 0
  imgc = np.copy(m)
  imgcc= np.copy(m)
  imgr = np.copy(m)
  img3= np.copy(m)

  color = np.zeros(np.max(labels)+1)
  r = np.zeros(np.max(labels)+1)
  x = np.zeros(np.max(labels)+1)
  y = np.zeros(np.max(labels)+1)
  i = 0

  for label in np.unique(labels):
      if label == 0:
          continue
      mask = np.zeros(pic1.shape, dtype="uint8")
      mask[labels == label] = 255
      cnts	 = cv2.findContours(mask.copy(),cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      cnt_len = cv2.arcLength(c, True)
      cnt = cv2.approxPolyDP(c, 0.06*cnt_len, True)
      k=cv2.isContourConvex(cnt)
      k=True
      if  (cv2.contourArea(c)>2):
        ((x[i], y[i]), r[i]) = cv2.minEnclosingCircle(c)

        if (r[i]>low)&(r[i]<up):
            cv2.circle(imgr, (int(x[i]), int(y[i])), int(r[i]), (255,0, 0), 1)
            color[i]=np.mean(pic1[c[:,:,1],c[:,:,0]])
            i +=1
  plt.imshow(imgr)
  plt.show
  color=color[r!=0]
  r = r[r!=0]
  x = x[x!=0]
  y = y[y!=0]
  size = np.size(r)
  r_total=np.concatenate((r_total, r), axis=0)        


  f=np.zeros((size))
  for h in range(size):
    for l in range(size):
      if h!=l:
        if np.sqrt((np.abs(x[h]-(x[l])))**2+np.abs(((y[h])-(y[l])))**2)<(0.95*r[h]+0.95*r[l]):
          # if np.sqrt((np.abs(x[h]-(x[l])))**2+np.abs(((y[h])-(y[l])))**2)<(0.5*r[h]+0.5*r[l]):
          #     if r[h]<r[l]:
          #       f[h]=1   
          #print(x[h],x[l],y[h],y[l],r[h],r[l])
          dif=pic1[int(y[h]),int(x[h])]-pic1[int(y[l]),int(x[l])]
          #if (pic1[int(y[h]),int(x[h])]>(pic1[int(y[l]),int(x[l])]))&(color[h]>(color[l])):
          if (pic1[int(y[h]),int(x[h])]>(pic1[int(y[l]),int(x[l])]+2)):
          #print(color[h],color[l])
            f[h]=1
          #else:
            #if r[h]<r[l]:
             # f[h]=1  
  for d in range(size):
    if f[d]==0:
        cv2.circle(img3, (int(x[d]), int(y[d])), int(r[d]), (255,0, 0), 1)  
  zeros[:,0:600,:]=imgr      
  cv2.putText(zeros, "rbc_count: {}".format(size), ( 225, 710),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255 , 255, 255), 1) 
  cv2.imwrite("C:/Users/Asus/Desktop/rbc_10x/"+file+file1,imgr)
rbc_count=np.size(r_total)*20/15
MCV=np.mean(r_total)
HCT=rbc_count*MCV*0.1
sigma=np.std(r_total)
RDW_CV=(sigma/MCV)*100
#Hgb
Hgb=HCT/3

#MCH[pg]
MCH=(Hgb*10)/rbc_count

#MCHC[g/dl]
MCHC=(Hgb*100)/HCT
print("rbc_count:",rbc_count)  