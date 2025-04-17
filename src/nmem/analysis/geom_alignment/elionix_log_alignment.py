# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:32:21 2023

@author: omedeiro
"""
import matplotlib.pyplot as plt
import numpy as np
logfile = "New schedule.log"


with open(logfile) as f:
    f = f.readlines()
    
# for line in f:
#     if "mark0" in line:
#         start = line.find("(")+1
#         end = line.find("[")
#         # print(line[start:end])
#         x1 = float(line[start:line.find(',')])
#         y1 = float(line[line.find(',')+1:line.find(')')])
        
#         x2 = float(line[line.find('(', line.find('(')+1)+1:line.find(',', line.find(',')+1)])
#         y2 = float(line[line.find(',', line.find(',')+1)+2:-7])

#         xdiff = (x1-x2)
#         ydiff = (y1-y2)
#         print([xdiff, ydiff])
        
diff_list = []
for line in f:
    if ".car\n" in line:
        pos1A = f[f.index(line)+17].split("\t")[5:7]
        pos1B = f[f.index(line)+24].split("\t")[5:7]
        pos1C = f[f.index(line)+31].split("\t")[5:7]
        pos1D = f[f.index(line)+38].split("\t")[5:7]
        
        pos2A = f[f.index(line)+45].split("\t")[5:7]
        pos2B = f[f.index(line)+52].split("\t")[5:7]
        pos2C = f[f.index(line)+59].split("\t")[5:7]
        pos2D = f[f.index(line)+66].split("\t")[5:7]
        pos_new = f[f.index(line)+14]
        print(f"pos1A {pos1A}")
        print(f"pos1B {pos1B}") 
        print(f"pos1C {pos1C}")
        print(f"pos1D {pos1D}")
        print(f"pos2A {pos2A}")
        print(f"pos2B {pos2B}")
        print(f"pos2C {pos2C}")
        print(f"pos2D {pos2D}")
        print(f"pos_new {pos_new}")
        if pos1A[0] != "0\n":
            diff1Ax = float(pos1A[0])-float(pos2A[0])
            diff1Ay = float(pos1A[1])-float(pos2A[1])
            
            diff1Bx = float(pos1B[0])-float(pos2B[0])
            diff1By = float(pos1B[1])-float(pos2B[1])

            diff1Cx = float(pos1C[0])-float(pos2C[0])
            diff1Cy = float(pos1C[1])-float(pos2C[1])
            
            diff1Dx = float(pos1D[0])-float(pos2D[0])
            diff1Dy = float(pos1D[1])-float(pos2D[1])
            
            diff_list.append(diff1Ax*1e6)
            diff_list.append(diff1Ay*1e6)
            diff_list.append(diff1Bx*1e6)
            diff_list.append(diff1By*1e6)
            diff_list.append(diff1Cx*1e6)
            diff_list.append(diff1Cy*1e6)
            diff_list.append(diff1Dx*1e6)
            diff_list.append(diff1Dy*1e6)

binwidth=1
n, bins, patches = plt.hist(x=diff_list, bins=range(int(np.floor(min(diff_list))), int(np.ceil(max(diff_list))) + binwidth, binwidth), edgecolor='black', color='#0504aa', alpha=0.5)

plt.ylabel('count')
plt.xlabel('alignment difference [nm]')
