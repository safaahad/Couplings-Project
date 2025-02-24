import numpy as np

dat = np.loadtxt('shift9_distance.csv)'
slip = dat[:,1]
theta = dat[:,2]
distn = dat[:,3]
print(distn)
