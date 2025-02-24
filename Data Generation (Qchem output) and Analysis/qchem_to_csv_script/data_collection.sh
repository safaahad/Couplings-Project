#!/bin/bash
for f in *.out
do 	
	shif=$(echo $f | sed 's/shift\(.*\)_slip\(.*\)_theta\(.*\).inp.out/\1/')
	slip=$(echo $f | sed 's/shift\(.*\)_slip\(.*\)_theta\(.*\).inp.out/\2/')
	theta=$(echo $f |sed 's/shift\(.*\)_slip\(.*\)_theta\(.*\).inp.out/\3/')
	ES1=$(grep 'Excited state   1: excitation energy (eV) =    *' $f) 
	arrayES1=( $ES1)
        ES2=$(grep 'Excited state   2: excitation energy (eV) =    *' $f)
        arrayES2=( $ES2)
        COU=$(grep '    1    2' $f) 
	arrayCOU=( $COU)
	echo "$shif,$slip,$theta,${arrayES1[7]},${arrayES2[7]},${arrayCOU[5]}" 
done
