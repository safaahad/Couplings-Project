import numpy as np
import math
import sys
import re 

myfile = sys.argv[1]
def get_min_dist(filename):
	#loading a file and pulling coordinates from it
	xyz = open(filename, "r")
	i = 0
	#open arrays for the coordinates
	at1 = []
	at2 = []
	x1 = []
	x2 = []
	y1 = []
	y2 = []
	z1 = []
	z2 = []
	#w = filename.split("_")
	#print(w[0])
	#w = np.char.split(filename, "_")	
	filename_str = re.findall(r"[+-]?\d+(?:\.\d+)?", filename )	

	for line in xyz:
		if i > 1 and i < 84:
		#split lines at breaks 
			at1.append((line.split()[0]))
			x1.append(float(line.split()[1]))
			y1.append(float(line.split()[2]))
			z1.append(float(line.split()[3]))

		elif i > 83 and i < 166:
			at2.append((line.split()[0]))
			x2.append(float(line.split()[1]))
			y2.append(float(line.split()[2]))
			z2.append(float(line.split()[3]))
		i = i + 1
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
	#closest_atom_indicies_molec1 = []
	#closest_atom_indicies_molec2 = []
	#step3: define for loop to iterate for the ith atom in molecule1 and jth atom in molecule2 
	for i in range(0, len(at1)): #need i and j here
		for j in range(0, len(at2)):
			#call distance function from the definition and use the indexing from the for loop 
			d = distance(x1[i], y1[i], z1[i], x2[j], y2[j], z2[j]) 
			#step 4: rewrite new min_distance value
			if d < min_distance:
				min_distance = d
				closest_atom_indicies_molec1 = i+1
				closest_atom_indicies_molec2 = j+83
				at_name = [at1[i], at2[j]]
				#array = [int(float(filename_str[0])), int(float(filename_str[1])),  int(float(filename_str[2])), min_distance, at1[i], closest_atom_indicies_molec1, at2[j], closest_atom_indicies_molec2]
	#print(array)
	#print(int(float(filename_str[0])), int(float(filename_str[1])),  int(float(filename_str[2])), min_distance, at1[i], closest_atom_indicies_molec1, at2[j], closest_atom_indicies_molec2, sep = ",")
	print(int(float(filename_str[0])), int(float(filename_str[1])),  int(float(filename_str[2])), min_distance, closest_atom_indicies_molec1, closest_atom_indicies_molec2, sep = ",")
	return min_distance, closest_atom_indicies_molec1, closest_atom_indicies_molec2


r,s,t = get_min_dist(myfile)
