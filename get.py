# -*- coding: utf-8 -*-

import cv2
from numpy import*
import random
import matplotlib.pyplot as plt
import sys

import copy

def read(filename):
    img = cv2.imread(filename,0)
    return img
def get_matlist(img):
    m,n = img.shape
  #  print m,n

    result = []
    row_counts8 = m/8
    column_counts8 = n/8

    for i in range(row_counts8):
        for j in range(column_counts8):
            temp8x8 = zeros((8,8))
            for k in range(8):
                for l in range(8):
                    temp8x8[k][l] = img[i*8+k][j*8+l]
            result.append(temp8x8)

  #  print result
    return result    

def get_next(row, column):
    sum = row + column
    if (row == 0 or row == 7) and column % 2 == 0:
        return row,column+1
    if (column == 0 or column == 7) and row % 2 == 1:
        return row+1,column
    if sum % 2 == 1:
        return row+1,column-1
    return row-1,column+1   
    

def LSB(original_map,secret_information):
    cot = 0
    m,n = original_map.shape    
    
    while(cot<len(secret_information)-1):
        cot +=1
        row = cot%m
        colum = cot/n        
        if (secret_information[cot]=='0'):
            if(original_map[row-1][colum]%2==1):
                original_map[row-1][colum] -= 1
        elif (secret_information[cot]=='1'):
            if(original_map[row-1][colum]%2==0):
                original_map[row-1][colum] += 1 
 #   print cot
    return original_map         

def relation(img):
    cot = 0
    m,n = img.shape
    start_m,start_n = 0,0
    next_m,next_n = get_next(start_m,start_n)
    while (next_m<m and next_n<n) :
        cot += abs((img[next_m][next_n]-img[start_n][start_n]))
        start_m,start_n = next_m,next_n
        next_m,next_n = get_next(next_m,next_n)
        
    return cot
    
def non_positive_rotation(img):#接收8x8
    jg = random.random()
    if(jg>0.5):
        return img
    else:
        next_m,next_n = 0,0
        while (next_m<8 and next_n<8) :
            if(img[next_m][next_n]%2==0 and img[next_m][next_n] != 0):
                img[next_m][next_n] -=1
            elif(img[next_m][next_n]%2==1 and img[next_m][next_n] != 255):
                img[next_m][next_n] +=1
            next_m,next_n = get_next(next_m,next_n)
        return img
        
def non_negative_rotation(img):#接收8x8
    jg = random.random()
    if(jg>0.5):
        return img
    else:
        next_m,next_n = 0,0
        while (next_m<8 and next_n<8) :
            if(img[next_m][next_n]%2==0):
                img[next_m][next_n] +=1
            elif(img[next_m][next_n]%2==1):
                img[next_m][next_n] -=1
            next_m,next_n = get_next(next_m,next_n)
        return img

def get_param(mat_list):
    relation_list=[]
    non_negro_relation_list=[]
    non_posro_relation_list=[]
    for i in range(len(mat_list)):
        temp = copy.deepcopy(mat_list[i])
        temp1=copy.deepcopy(temp)
        temp2=copy.deepcopy(temp)
        relation_list.append(relation(temp))
        non_posro_relation_list.append(relation(non_positive_rotation(temp2)))
        non_negro_relation_list.append(relation(non_negative_rotation(temp1)))        

    sm = 0.0
    rm = 0.0
    s_m=0.0
    r_m=0.0
    for i in range(len(mat_list)):
        
        if((relation_list[i]-non_negro_relation_list[i])>0):
            sm+=1
        elif((relation_list[i]-non_negro_relation_list[i])<0):
            rm+=1
            
        if((relation_list[i]-non_posro_relation_list[i])>0):
            s_m+=1
        elif((relation_list[i]-non_posro_relation_list[i])<0):
            r_m+=1
    sm = sm/len(relation_list)
    rm = rm/len(relation_list)
    s_m = s_m/len(relation_list)
    r_m = r_m/len(relation_list)
    print "sm,s_m,rm,r_m"
    print sm,s_m,rm,r_m
    return sm,s_m,rm,r_m
def getimplant(img,secret_information,percentage):
    newlen = int(len(secret_information)*percentage)
    temp_info = secret_information[0:newlen]  
    
    tempimg = copy.deepcopy(img)
    implant_img = LSB(tempimg,temp_info)
    implant_mat_list = get_matlist(implant_img)
    
    return get_param(implant_mat_list)
#
photofilename = sys.argv[1]
img = read(photofilename)
original_mat_list = get_matlist(img)

m, n = img.shape

secret_information = ""
for i in xrange(m * n):
    jg = random.random()
    if(jg>0.5):
        secret_information +='0'
    else:
        secret_information +='1'
print "begin"
result_list = []
ratio = []
for i in range(11):
    cur_ratio = i * 0.1
    ratio.append(cur_ratio)
    xx = copy.deepcopy(img)
    re = getimplant(xx,secret_information,cur_ratio)
    result_list.append(re)  
print "end function"    
sm = []
s_m = []
rm = []
r_m = []
for i in result_list:
    sm.append(i[0])
    s_m.append(i[1])
    rm.append(i[2])
    r_m.append(i[3])
print result_list
print "end"
plt.plot(ratio, sm, marker='o')#  mec='r', mfc='w',label=u'y=x^2曲线图')
plt.plot(ratio, s_m, marker='*') #ms=10,label=u'y=x^3曲线图')
plt.plot(ratio, rm, marker='^') #mec='r', mfc='w',label=u'y=x^2曲线图')
plt.plot(ratio, r_m, marker="+") #, ms=10,label=u'y=x^3曲线图')
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
# plt.xlabel(u"time(s)邻居") #X轴标签
# plt.ylabel("RMSE") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()


