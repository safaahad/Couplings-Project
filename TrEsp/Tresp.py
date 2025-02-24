#!/usr/bin/env python
# coding: utf-8

# In[529]:


#MG DIMER


dipolelengthsMG=[Mg1, Mg2]
PorphDipLength=[8.7870e-18, 9.7337e-18]


# In[2]:


import os,sys
sys.path.append('./misc/lib/python3.7/site-packages')

import math, time, os, requests, re
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import Javascript, display
from os import listdir
from os.path import join, isfile
import glob

np.set_printoptions(precision=1000)
np.set_printoptions(suppress=True)


# In[6]:


import pandas as pd
import glob, os, math, numpy, fileinput, subprocess

data=np.loadtxt('test.csv', delimiter=',')
# dfs=list()
# fname = os.path.join("./",
#                      "shift8.0_slip9_theta0.csv")
# data = pd.read_csv(fname, header=None)
data1=data[0:82]
data2=data[82:164]

AtNames=[data[0]]
ZnMon1_AtNames=[data1[0]]
ZnMon2_AtNames=[data2[0]]

#Monomer1
x1=[data1[1]]
y1=[data1[2]]
z1=[data1[3]]

#Monomer2
x2=[data2[1]]
y2=[data2[2]]
z2=[data2[3]]


PorphAtNames=list()
PorphAtNames.append(ZnMon1_AtNames)
PorphAtNames.append(ZnMon2_AtNames)

PorphQ10=list()
PorphQ10.append(ZnMon1_Q10)
PorphQ10.append(ZnMon2_Q10)

PorphNames = ['ZnMon1', 'ZnMon2']
PorphResNames = ['ZnMon1', 'ZnMon2']

#dipolelengthsZn = [Zn1, Zn2]
PorphDipLength = [8.7413e-18, 9.6683e-18] #I got these values from the optimized geometry of each monomer Debye

h = 6.62607015e-34 #J*s
c = 2.998e10 #cm/s
eo = 4.80320451e-10 # esu
    
Erg2J = 1.0e-7
ang2cm = 1.0e-8

#Transition chargesdat1
ZnMon1_AtNames = ['Zn', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'N', 'C', 'C', 'H', 'C', 'H', 'C', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'C', 'O', 'O', 'N', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H','C', 'H', 'C', 'H', 'H', 'N', 'C', 'C', 'C', 'C', 'C','H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H', 'N', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'C', 'O', 'C', 'H', 'C', 'O', 'O', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
ZnMon1_Q10 = np.array([-0.0062,0.1184,-0.0297,0.0008,-0.1864,0.0176,0.0582,-0.0020,0.0128,-0.0874,-0.0527,0.0100,0.0184,-0.0010,0.0975,-0.0187,0.0069,0.0124,0.0003,0.0121,-0.0044,-0.0006,0.0119,-0.0066,-0.0048,-0.0008,0.0019,-0.0026,-0.0892,0.0663,0.0049,0.0148,0.1575,-0.0043,0.0069,0.0088,0.0046,-0.0243,0.0073,0.0285,-0.0036,0.0018,-0.0261,0.1611,-0.0214, 0.0285, -0.0584, -0.0065, 0.0010, 0.0026, 0.0005, -0.0054, 0.0045, 0.0018, 0.0035, -0.0018, 0.0025, 0.0012, 0.0923, -0.0955, -0.0198, -0.0346, -0.1194, 0.0136, -0.0156, -0.0041, -0.0146, 0.0217, -0.0291,-0.0544,0.0091,0.0257,-0.0082,-0.0107,-0.0004,0.0031,-0.0030,0.0013,0.0006,0.0000,-0.0006,-0.0001])
ZnMon1_Q10 = ZnMon1_Q10*4.80326e-10*1e-3 # cgs base units for charge: cm^3/2 g^1/2 s^−1

ZnMon2_AtNames = ['Zn', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'N', 'C', 'C', 'H', 'C', 'H', 'C', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'C', 'O', 'O', 'N', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H','C', 'H', 'C', 'H', 'H', 'N', 'C', 'C', 'C', 'C', 'C','H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H', 'N', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'C', 'O', 'C', 'H', 'C', 'O', 'O', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
ZnMon2_Q10 = np.array([-0.0014,0.1206,-0.0347,0.0011,-0.1884,0.0175,0.0455,-0.0009,0.0094,-0.0861,-0.0513,0.0107,0.0145,-0.0003,0.1074,-0.0147,0.0054,0.0114,-0.0003,-0.0027,-0.0009,0.0041,0.0210,-0.0080,-0.0057,-0.0055,-0.0006,0.0032,-0.0940,0.0678,0.0045,0.0140,0.1588,-0.0055,0.0073,0.0091,0.0047,-0.0225,0.0067,0.0272,-0.0034,0.0018,-0.0263,0.1592,-0.0149,0.0204,-0.0503,-0.0070,0.0007,0.0025,0.0006,-0.0026,0.0041,0.0023,0.0001,-0.0012,0.0032,0.0021,0.0839,-0.0828,-0.0233,-0.0377,-0.1126,0.0176,-0.0156,-0.0163,-0.0053,0.0279,-0.0293,-0.0694,0.0128,0.0271,-0.0068,-0.0107,0.0046,0.0018,-0.0038,-0.0006,-0.0032,0.0018,-0.0001,0.0004])
ZnMon2_Q10 = ZnMon2_Q10*4.80326e-10*1e-3


# In[9]:


Coupling=0
for atm in range(0, len(data1)):
    for atn in range(0, len(data2)):
        Rmn = (data1[atm,:] - data2[atn,:])*1.0e-8
        rmn = np.linalg.norm(Rmn) 
        Coupling+= (((ZnMon1_Q10[atm]*ZnMon2_Q10[atn])/(rmn))*(1e-7))/(h*c)
print(Coupling)
#CoupMat = CoupMat + np.transpose(CoupMat)


# In[ ]:




