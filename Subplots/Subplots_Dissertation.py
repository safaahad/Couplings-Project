#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




# maskmg = False
# maskzn = False
#maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
#maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = 1 > data3_np_mg
maskzn =1 > data6_np_zn
vmax=1500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 3.0', y=1.02)
plt.show()


# In[41]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




# maskmg = False
# maskzn = False
#maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
#maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = 1 > data3_np_mg
maskzn =1 > data6_np_zn
vmax=1500

# Set font scale
sns.set(font_scale=2.5)

# Create subplots with equal size
fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(nrows=2, ncols=2, figsize=(40, 30), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)
ax2.yaxis.tick_right()
ax2.yaxis.set_ticks_position('none')

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')
ax5.yaxis.tick_right()
ax5.yaxis.set_ticks_position('none')

# Adjust layout to make subplots closer
plt.subplots_adjust(hspace=0.03, wspace=0.05)

# Set aspect ratio to be equal
for ax in [ax1, ax2, ax4, ax5]:
    ax.set_aspect('equal', adjustable='box')
    
cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.76])  # [left, bottom, width, height]
fig.colorbar(ax1.collections[0], cax=cbar_ax)

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 3.0', y=0.97)
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######

# maskmg = False
maskmg = 1 > data3_np_mg
maskzn =1 > data6_np_zn
vmax=1500

    
y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']
sns.set(font_scale=3.5)

# Create subplots with equal aspect ratio
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(40, 40), constrained_layout=True)

# Plot the heatmaps
sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=True)
sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax3, cbar=False)
sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=True)

# Set titles and labels
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.02)
ax1.set_ylabel('Slip ($\AA$)')
ax1.set_xlabel('Rotation ($\Theta$)')

ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)', y=1.02)
ax2.set_ylabel('Slip ($\AA$)')
ax2.set_xlabel('Rotation ($\Theta$)')

ax3.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.02)
ax3.set_ylabel('Slip ($\AA$)')
ax3.set_xlabel('Rotation ($\Theta$)')

ax4.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=1.02)
ax4.set_ylabel('Slip ($\AA$)')
ax4.set_xlabel('Rotation ($\Theta$)')

plt.suptitle('Shift 3.0, Unmasked', y=1.02)

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######

maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=1000

    
y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']
sns.set(font_scale=3.5)

# Create subplots with equal aspect ratio
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(40, 30), constrained_layout=True)

# Plot the heatmaps
sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=True)
sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax3, cbar=False)
sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=True)

# Set titles and labels
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.16)
ax1.set_ylabel('Slip ($\AA$)')
ax1.set_xlabel('Rotation ($\Theta$)')
ax1.xaxis.set_label_position('top')
ax1.set_xticklabels(x_axis_labels, rotation=90)
ax1.xaxis.tick_top()

ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)', y=1.02)
ax2.set_ylabel('Slip ($\AA$)')
ax2.set_xlabel('Rotation ($\Theta$)')
ax2.xaxis.set_label_position('top')
ax2.set_xticklabels(x_axis_labels, rotation=90)
ax1.xaxis.tick_top()

ax3.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.02)
ax3.set_ylabel('Slip ($\AA$)')
ax3.set_xlabel('Rotation ($\Theta$)')

ax4.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=1.02)
ax4.set_ylabel('Slip ($\AA$)')
ax4.set_xlabel('Rotation ($\Theta$)')

plt.suptitle('Shift 3.0, Unmasked', y=1.02)

plt.show()


# In[50]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = (0.5 > data3_np_mg)
maskzn = (0.5 > data6_np_zn) 
vmax=1000






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 3.5', y=1.02)
plt.show()


# In[53]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append((float(contents[n].split("'")[0].split(',')[3]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append((float(contents4[n].split("'")[0].split(',')[3]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = (0.5 > data3_np_mg)
maskzn = (0.5 > data6_np_zn) 
vmax=19000






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP ES1 ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP ES1 ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 3.5', y=1.02)
plt.show()


# In[382]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift3.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift3.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift3.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift3.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift3.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift3.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=750






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 3.5', y=1.02)
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
#maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
#maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = 1.0 > data3_np_mg
maskzn = 1.0 > data6_np_zn
vmax=550






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.0', y=1.02)
plt.show()


# In[62]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = (1.0 > data3_np_mg)
maskzn = (1.0 > data6_np_zn) 
vmax=1000






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.0', y=1.02)
plt.show()


# In[55]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = (0.5 > data3_np_mg) 
maskzn = (0.5 > data6_np_zn)
vmax=1000






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.0', y=1.02)
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
maskmg = (0.5 > data3_np_mg) 
maskzn = (0.5 > data6_np_zn)
vmax=1000






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2], [ax4, ax5]) = plt.subplots(ncols=2, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=True)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

# mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
# ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
# ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
# ax3.set_yticklabels(y_axis_labels, rotation=360)
# ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=True)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

# zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
# ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
# ax6.set_xticklabels(x_axis_labels, rotation=90)
# ax6.set_yticklabels(y_axis_labels, rotation=360)
# ax6.tick_params(axis="both", direction="in", pad=15)
# ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.0, Masked', y=1.02)
plt.show()


# In[ ]:





# In[380]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 


###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=550






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.0', y=1.02)
plt.show()


# In[377]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######





# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.5', y=1.02)
plt.show()


# In[63]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######





# maskmg = False
# maskzn = False
maskmg = (1.0 > data3_np_mg) 
maskzn = (1.0 > data6_np_zn)
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.5', y=1.02)
plt.show()


# In[378]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift4.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift4.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift4.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift4.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift4.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift4.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######





maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 4.5', y=1.02)
plt.show()


# In[375]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 5.0', y=1.02)
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2], [ax4, ax5]) = plt.subplots(ncols=2, nrows=2, figsize=(30,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

# fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
# fig.subplots_adjust(wspace=0.01)
# fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer TD-DFT Coupling', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

# mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
# ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
# ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
# ax3.set_yticklabels(y_axis_labels, rotation=360)
# ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer TD-DFT Coupling', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

# zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
# ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
# ax6.set_xticklabels(x_axis_labels, rotation=90)
# ax6.set_yticklabels(y_axis_labels, rotation=360)
# ax6.tick_params(axis="both", direction="in", pad=15)
# ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
#zncou.figure.suptitle('Shift 5.0', y=1.02)
plt.show()


# In[376]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=500






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 5.0', y=1.02)
plt.show()


# In[373]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift5.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift5.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift5.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift5.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift5.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift5.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=300






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 5.5', y=1.02)
plt.show()


# In[374]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift5.5.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift5.5_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift5.5_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift5.5.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift5.5_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift5.5_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######




maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=300






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 5.5', y=1.02)
plt.show()


# In[371]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift6.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift6_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift6_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift6.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift6_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift6_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 

vmax=250






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 6.0', y=1.02)
plt.show()


# In[372]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift6.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift6_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift6_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift6.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift6_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift6_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 

vmax=250






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 6.0', y=1.02)
plt.show()


# In[370]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift7.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift7_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift7_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift7.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift7_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift7_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=250






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 7.0', y=1.02)
plt.show()


# In[369]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift7.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift7_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift7_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift7.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift7_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift7_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=250






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 7.0', y=1.02)
plt.show()


# In[367]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift8.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift8_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift8_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift8.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift8_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift8_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=200






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 8.0', y=1.02)
plt.show()


# In[368]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift8.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift8_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift8_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift8.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift8_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift8_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=200






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 8.0', y=1.02)
plt.show()


# In[365]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift9.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift9_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift9_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift9.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift9_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift9_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=125






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 9.0', y=1.02)
plt.show()


# In[366]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift9.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift9_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift9_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift9.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift9_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift9_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=125






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 9.0', y=1.02)
plt.show()


# In[364]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift10.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift10_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift10_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift10.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift10_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift10_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=100






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

# plt.colorbar(ax1.get_children()[0], ax = [ax1])
zncou.figure.suptitle('Shift 10.0', y=1.02)
plt.show()


# In[363]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift10.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift10_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift10_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift10.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift10_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift10_distance_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=100






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

zncou.figure.suptitle('Shift 10.0', y=1.02)
plt.show()


# In[387]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift11.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift11_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift11_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift11.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift11_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift11_distances_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=75






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

zncou.figure.suptitle('Shift 11.0', y=1.02)
plt.show()


# In[389]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift11.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift11_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift11_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift11.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift11_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift11_distances_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=100






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

zncou.figure.suptitle('Shift 11.0', y=1.02)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift12.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift12_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift12_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift12.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift12_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift12_distances_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



# maskmg = False
# maskzn = False
maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=100






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

zncou.figure.suptitle('Shift 12.0', y=1.02)
plt.show()


# In[390]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

###mg qm coupling###
mgdata = open('mg_shift12.txt', "r")
contents=mgdata.read().split("\n")

x = [] #contents[:,0] shift
y = [] #contents[:,1] slip
z = [] #contents[:,2] theta
a = [] #contents[:,5] cou

for n in range (0,len(contents)):
    x.append(float(contents[n].split("'")[0].split(',')[0])) #shift
    y.append(float(contents[n].split("'")[0].split(',')[1])) #slip
    z.append(float(contents[n].split("'")[0].split(',')[2])) #rotation
    a.append(abs(float(contents[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data_mat_mg = []
for n in range(25):
    data_mat_mg.append([])
    for m in range(36):
        data_mat_mg[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data_mat_mg[i][j]=a[k]
                
data_np_mg = np.array(data_mat_mg)
###mg tresp data###
mgtrespdata = open('Mg_shift12_Tresp_cou.txt', "r")
contents2=mgtrespdata.read().split("\n")

x2 = [] #contents2[:,0] shift
y2 = [] #contents2[:,1] slip
z2 = [] #contents2[:,2] theta
a2 = [] #contents2[:,5] cou

for n in range (0,len(contents2)):
    x2.append(float(contents2[n].split("'")[0].split(',')[0])) #shift
    y2.append(float(contents2[n].split("'")[0].split(',')[1])) #slip
    z2.append(float(contents2[n].split("'")[0].split(',')[2])) #rotation
    a2.append(abs(float(contents2[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data2_mat_mg = []
for n in range(25):
    data2_mat_mg.append([])
    for m in range(36):
        data2_mat_mg[n].append(0)
# numpy matrix initialization function call for two rows, 36 columns, all zero
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y2[k]==(i-12) and z2[k]==(10*j)):
                data2_mat_mg[i][j]=a2[k]
data2_np_mg = np.array(data2_mat_mg) 

###mg shortest distance###

mgdisdata = open('Mg_shift12_distance.txt', "r")
contents3=mgdisdata.read().split("\n")

x3 = [] #contents[:,0] shift
y3 = [] #contents[:,1] slip
z3 = [] #contents[:,2] theta
a3 = [] #contents[:,5] cou

for n in range (0,len(contents3)):
    x3.append(float(contents3[n].split("'")[0].split(',')[0])) #shift
    y3.append(float(contents3[n].split("'")[0].split(',')[1])) #slip
    z3.append(float(contents3[n].split("'")[0].split(',')[2])) #rotation
    a3.append(float(contents3[n].split("'")[0].split(',')[3])) #shortestdistance
    
data3_mat_mg = []
for n in range(25):
    data3_mat_mg.append([])
    for m in range(36):
        data3_mat_mg[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y3[k]==(i-12) and z3[k]==(10*j)):
                data3_mat_mg[i][j]=a3[k]
data3_np_mg = np.array(data3_mat_mg) 



###############zn data#######################
        

###zn qm coupling###
zndata = open('zn_shift12.txt', "r")
contents4=zndata.read().split("\n")

x4 = [] #contents[:,0] shift
y4 = [] #contents[:,1] slip
z4 = [] #contents[:,2] theta
a4 = [] #contents[:,5] cou

for n in range (0,len(contents4)):
    x4.append(float(contents4[n].split("'")[0].split(',')[0])) #shift
    y4.append(float(contents4[n].split("'")[0].split(',')[1])) #slip
    z4.append(float(contents4[n].split("'")[0].split(',')[2])) #rotation
    a4.append(abs(float(contents4[n].split("'")[0].split(',')[5]))*8065.73) #qmcoupling
    
data4_mat_zn = []
for n in range(25):
    data4_mat_zn.append([])
    for m in range(36):
        data4_mat_zn[n].append(0)

for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y[k]==(i-12) and z[k]==(10*j)):
                data4_mat_zn[i][j]=a[k]
                
data4_np_zn = np.array(data4_mat_zn)
###zn tresp data###
zntrespdata = open('Zn_shift12_Tresp_cou.txt', "r")
contents5=zntrespdata.read().split("\n")

x5 = [] #contents2[:,0] shift
y5 = [] #contents2[:,1] slip
z5 = [] #contents2[:,2] theta
a5 = [] #contents2[:,5] cou

for n in range (0,len(contents5)):
    x5.append(float(contents5[n].split("'")[0].split(',')[0])) #shift
    y5.append(float(contents5[n].split("'")[0].split(',')[1])) #slip
    z5.append(float(contents5[n].split("'")[0].split(',')[2])) #rotation
    a5.append(abs(float(contents5[n].split("'")[0].split(',')[3]))) #trespcoupling
    
data5_mat_zn = []
for n in range(25):
    data5_mat_zn.append([])
    for m in range(36):
        data5_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y5[k]==(i-12) and z5[k]==(10*j)):
                data5_mat_zn[i][j]=a5[k]
data5_np_zn = np.array(data5_mat_zn) 

###zn shortest distance###

zndisdata = open('Zn_shift12_distances_full.txt', "r")
contents6=zndisdata.read().split("\n")

x6 = [] #contents[:,0] shift
y6 = [] #contents[:,1] slip
z6 = [] #contents[:,2] theta
a6 = [] #contents[:,5] cou

for n in range (0,len(contents6)):
    x6.append(float(contents6[n].split("'")[0].split(',')[0])) #shift
    y6.append(float(contents6[n].split("'")[0].split(',')[1])) #slip
    z6.append(float(contents6[n].split("'")[0].split(',')[2])) #rotation
    a6.append(float(contents6[n].split("'")[0].split(',')[3])) #shortestdistance
    
data6_mat_zn = []
for n in range(25):
    data6_mat_zn.append([])
    for m in range(36):
        data6_mat_zn[n].append(0)
for i in range(25):
    for j in range(36):
        for k in range(900):
            if (y6[k]==(i-12) and z6[k]==(10*j)):
                data6_mat_zn[i][j]=a6[k]
data6_np_zn = np.array(data6_mat_zn) 


####mask#######



maskmg = False
maskzn = False
# maskmg = (0.5 > data3_np_mg) | ( data3_np_mg > 1.0) 
# maskzn = (0.5 > data6_np_zn) | ( data6_np_zn > 1.0) 
vmax=100






##################
sns.set(font_scale = 3.5)

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(50,30),sharey=True)
fig.subplots_adjust(wspace=0.01)
fig = plt.figure()

y_axis_labels = ['-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
x_axis_labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350', '360']

mgcou = sns.heatmap(data_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax1, cbar=False)
ax1.set_title('Mg-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=1.12)
ax1.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax1.set_yticklabels(y_axis_labels, rotation=360)
ax1.tick_params(axis="both", direction="in", pad=15)
ax1.set_ylabel('Slip ($\AA$)')

mgtresp = sns.heatmap(data2_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax2, cbar=False)
ax2.set_title('Mg-Dimer TrEsp Coupling ($cm^{-1}$)',y=1.12)
ax2.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax2.set_yticklabels(y_axis_labels, rotation=360)
ax2.tick_params(axis="both", direction="in", pad=15)

mgdist = sns.heatmap(data3_np_mg, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskmg, vmax=vmax, ax=ax3, cbar=True)
ax3.set_title('Mg-Dimer Shortest Distances', y=1.12)
ax3.set_xticklabels(x_axis_labels, rotation=90, y=1.12)
ax3.set_yticklabels(y_axis_labels, rotation=360)
ax3.tick_params(axis="both", direction="in", pad=15)

zncou = sns.heatmap(data4_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax4, cbar=False)
ax4.set_title('Zn-Dimer 6-G31*/B3LYP Coupling ($cm^{-1}$)', y=-0.25)
ax4.set_xticklabels(x_axis_labels, rotation=90)
ax4.set_yticklabels(y_axis_labels, rotation=360)
ax4.tick_params(axis="both", direction="in", pad=15)
ax4.set_xlabel('Rotation ($\Theta$)')
ax4.set_ylabel('Slip ($\AA$)')

zntresp = sns.heatmap(data5_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax5, cbar=False)
ax5.set_title('Zn-Dimer TrEsp Coupling ($cm^{-1}$)', y=-0.25)
ax5.set_xticklabels(x_axis_labels, rotation=90)
ax5.set_yticklabels(y_axis_labels, rotation=360)
ax5.tick_params(axis="both", direction="in", pad=15)
ax5.set_xlabel('Rotation ($\Theta$)')

zndist = sns.heatmap(data6_np_zn, cmap="rainbow", xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=maskzn, vmax=vmax, ax=ax6, cbar=True)
ax6.set_title('Zn-Dimer Shortest Distances', y=-0.25)
ax6.set_xticklabels(x_axis_labels, rotation=90)
ax6.set_yticklabels(y_axis_labels, rotation=360)
ax6.tick_params(axis="both", direction="in", pad=15)
ax6.set_xlabel('Rotation ($\Theta$)')

zncou.figure.suptitle('Shift 12.0', y=1.02)
plt.show()


# In[ ]:




