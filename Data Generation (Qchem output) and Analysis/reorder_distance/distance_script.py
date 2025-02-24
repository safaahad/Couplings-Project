import numpy as np
import math

#loading a file and pulling coordinates from it
filename = "dimer.csv"
xyz = open(filename, "r")

#open arrays for the coordinates
at = []
x = []
y = []
z = []

for line in xyz:
	#split lines at breaks 
	row = line.split()
	at.append(row[0])
	x.append(row[1])
	y.append(row[2])
	z.append(row[3])

#molecule1
at1 = np.array(at[0:82])
x1 = np.array(x[0:82], dtype=float)
y1 = np.array(y[0:82], dtype=float)
z1 = np.array(z[0:82], dtype=float)

#molecule2
at2 = np.array(at[83:165])
x2 = np.array(x[83:164], dtype=float)
y2 = np.array(y[83:164], dtype=float)
z2 = np.array(z[83:164], dtype=float)
xyz.close()

#how to calculate minimum distance
#step 1: set minimum distance to arbitrary value
min_distance = 10000 #angstroms

#step2: define a distance function
def distance(x1, y1, z1, x2, y2, z2):
	#we needed the float() because for some reason the array were not appending numeric values and were appending <u32 or <u11 instead of floats
	distance =  math.sqrt((float(x2)-float(x1))**2+(float(y2)-float(y1))**2+(float(z2)-float(z1))**2) #don't need i or j here
	return distance

#set array for closest indicies
closest_atom_indicies = []

#step3: define for loop to iterate for the ith atom in molecule1 and jth atom in molecule2 
for i in range(0, len(at1)): #need i and j here
	for j in range(0, len(at2)):
		#call distance function from the definition and use the indexing from the for loop 
		d = distance(x1[i], y1[i], z[i], x2[j], y2[j], z2[j]) 
		#step 4: rewrite new min_distance value
		if d < min_distance:
			min_distance = d
			closest_atom_indicies = [i, j+82]
			at_name = [at[i], at[j]]
print("The minimum distance between the two molecules is", min_distance, "angstroms.")
print("The indices for these atoms are", closest_atom_indicies)
print(at_name)
