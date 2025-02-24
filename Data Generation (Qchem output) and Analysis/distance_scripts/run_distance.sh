#!/bin/bash
module load anaconda
for f in *.inp
do
	python updated_distance_script.py $f
done
