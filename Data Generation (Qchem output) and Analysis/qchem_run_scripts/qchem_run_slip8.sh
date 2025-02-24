#!/bin/bash
FILES="shift*_slip8*.inp"
for f in $FILES
do
	source /depot/lslipche/etc/bashrc
	qc53 -q standby -w 04:00:00 -ccp 24 $f
done
