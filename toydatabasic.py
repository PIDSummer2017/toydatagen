import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy import optimize

def _image(array):
    fig,ax = plt.subplots(figsize=(28,28),facecolor='w')
    plt.imshow(array,interpolation = 'nearest')
    # plt.show()

def _choose_triangle(x, y,array):
    z = int(random.uniform(3,4))
    if (z < 3 or z >=4) and array[x,y] < 10:
        array[x,y] = 0
        return
    if 3 <= z < 4:
        if 3 <= x <= len(array[0])-3:
            if 4 <= y <= len(array[1]) - 4:
                array[x, y] = 180 + array[x, y]
                array[x-1:x+1, y-1] = 180 + array[x-1:x+1, y-1]
                array[x-2:x+2, y-2] = 180 + array[x-2:x+2, y-2]
                array[x-3:x, y-3] = 180 + array[x-3:x, y-3]
                array[x-4:x, y-4] = 180 + array[x-4:x, y-4]
                array[x:x+3, y-3] = 180 + array[x:x+3, y-3]
                array[x:x+4, y-4] = 180 + array[x:x+4, y-4]


def _choose_rectangle(x, y, array):
    z = random.uniform(0,1)
    if z > 1 and array[x, y] <10:
        array[x,y] = 0
        returm
    if 0 < z <= 1:
        if x+5 <= len(array[0]):
            if y+5 <= len(array[1]):
                array[x:x+5, y:y+5] = 0
                points = [(x, y), (x+5, y), (x, y+5), (x+5, y+5)]
                start_pt, end_pt = min(points), max(points)
                array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1] = 180 + array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1]

def _choose_horizontal(x, y, array):
    z = random.uniform(0,1)
    if 0 < z <= 1:
         if y+5 <= len(array[1]):
             points = [(x, y), (x, y+5)]
             array[x, y:y+5] = 180 + array[x, y:y+5]
              #print 1

def _choose_vertical(x, y, array):
    z = random.uniform(1,2)
    if (z > 2 or z <1) and array[x, y] < 10:
        array[x,y]  = 0
        return
    if 1 < z <= 2:
        if x+5 <= len(array[0]):
            points = [(x, y), (x+5, y)]
            array[x:x+5, y] = 180 + array[x:x+5, y]
