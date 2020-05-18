# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:59:05 2019

Creates a raster plot from the raster data. 

@author: logan
"""

from neuron import h 
from matplotlib import pyplot as plt
import numpy as np  
#import config
#import bazh_net_v4 
#from cell_classes import Cell, PyrCell, InhCell, RECell, TCCell

Npyr = 500 
Ninh = 100 
Nre = 100 
Ntc = 100  
pyrList = []
inhList = [] 
reList = [] 
tcList = [] 
pyrTime = [] 
inhTime = [] 
reTime = [] 
tcTime = [] 

raster_data = np.loadtxt("raster_nhost=6.txt") 

for i_len in range(len(raster_data)):
    if raster_data[i_len][1] < Npyr:
        pyrList.append(raster_data[i_len][1])
        pyrTime.append(raster_data[i_len][0])
    if raster_data[i_len][1] >= Npyr and raster_data[i_len][1] < (Npyr+Ninh):
        inhList.append(raster_data[i_len][1])
        inhTime.append(raster_data[i_len][0])
    if raster_data[i_len][1] >= (Npyr+Ninh) and raster_data[i_len][1] < (Npyr+Ninh+Nre):
        reList.append(raster_data[i_len][1])
        reTime.append(raster_data[i_len][0])
    if raster_data[i_len][1] >= (Npyr+Ninh+Nre) and raster_data[i_len][1] < (Npyr+Ninh+Nre+Ntc):
        tcList.append(raster_data[i_len][1])
        tcTime.append(raster_data[i_len][0])
    
plt.figure() 
plt.scatter(pyrTime, pyrList, marker='o', s=5, color='red')
plt.scatter(inhTime, inhList, marker='o', s=5, color='blue')
plt.scatter(reTime, reList, marker='o', s=5, color='green')
plt.scatter(tcTime, tcList, marker='o', s=5, color='orange')

        
