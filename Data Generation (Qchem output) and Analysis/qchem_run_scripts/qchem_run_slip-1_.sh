#!/bin/bash
FILES="shift*_slip-1*.inp"
for f in $FILES
do
	source /depot/lslipche/etc/bashrc
	qc53 -q standby -w 04:00:00 -ccp 24 $f
done
