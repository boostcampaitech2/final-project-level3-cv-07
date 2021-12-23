from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import cv2
from scipy.signal import find_peaks


import time
import datetime
# 이미지를 필요한 부분만 cutting 해줌
def cutting_img(img, box):
    time_start = time.monotonic()
    # 변환 전 4개 좌표 
    pts1 = np.float32(box)

    width = 180 # 두 좌우 거리간의 최대값이 서류의 폭
    height = 32 # 나오는 글씨의 높이
    
    # 변환 후 4개 좌표
    pts2 = np.float32([[0,0], [width-1,0], 
                        [width-1,height-1], [0,height-1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(img, mtrx, (width, height))
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'cutting_img : {running_time:.2f}s')  
    return result
# 이미지를 받으면 색이 많은것, 2번째로 많은 것을 return
def Find_MuchColor(img):
    time_start = time.monotonic()
    clt = KMeans(n_clusters=3)

    k_cluster = clt.fit(img.reshape(-1, 3))
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = sorted(perc.items(), key = lambda item: item[1], reverse = True)

    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'Find_MuchColor : {running_time:.2f}s')
    return [k_cluster.cluster_centers_[perc[0][0]].tolist(), k_cluster.cluster_centers_[perc[1][0]].tolist()]

def calc_color(img,idx):
    hist = cv2.calcHist([img],[idx],None,[256],[0,257])
    peaks,_=find_peaks(hist.flatten(),distance=16)
    temp=hist[peaks].flatten()
    first_idx=np.argmax(temp)
    temp[first_idx]=0
    second_idx=np.argmax(temp)
    return int(peaks[first_idx]),int(peaks[second_idx])

def calc_rgb(img):
    first_r,second_r=calc_color(img,0)
    first_g,second_g=calc_color(img,1)
    first_b,second_b=calc_color(img,2)
    return [[first_r,first_g,first_b],[second_r,second_g,second_b]]


# 이미지와 박스 리스트를 받으면 index 순서에 맞추어 배경, 글자색순으로 return
def color_list(img, box_list):
    colors_in_img = []
    for box in box_list:
        cutted_img = cutting_img(img,box)
        # colors = Find_MuchColor(cutted_img)
        colors=calc_rgb(cutted_img)####################
        colors_in_img.append(colors)
    return colors_in_img

def color_list_clustering(img, box_list):
    colors_in_img = []
    for box in box_list:
        cutted_img = cutting_img(img,box)
        colors = Find_MuchColor(cutted_img)#####################
        # colors=calc_rgb(cutted_img)
        colors_in_img.append(colors)
    return colors_in_img