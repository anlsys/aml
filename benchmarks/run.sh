#!/bin/sh
# make sure we stop on errors, and log everything
set -e
set -x
set -u
DATE=$(date +%Y%m%d.%H%M%S)
LOGFILE="$DATE.log"
DELIMITER="################################################################################"

# have stdout redirected to our LOGFILE
exec > "$LOGFILE"

# remember this script in the log
printf '%s\n%s\n\n' "$DELIMITER" "$DELIMITER"
cat "$0"
printf '\n%s\n' "$DELIMITER"

# Setup environment
echo "Start: $DATE"
export PATH=../../benchmarks/:$PATH

# Gather some info on the system
uname -a
cat /proc/cmdline
numactl --hardware
git describe --always
env
printf '\n%s\n' "$DELIMITER"
printf "Parameters: %s" "$*"

PROG=$1
FAST_MEM=$2
SLOW_MEM=$3
REPEATS=$4
TH_MIN=$5
TH_STEP=$6
TH_MAX=$7
SZ_MIN=$8
SZ_MAX=$9
SZ_STEP=${10}

for th in $(seq $TH_MIN $TH_STEP $TH_MAX)
do
	export OMP_NUM_THREADS=$th
	export OMP_PLACES=cores
	export OMP_PROC_BIND=spread
	for sz in $(seq $SZ_MIN $SZ_STEP $SZ_MAX)
	do
		for times in $(seq 1 $REPEATS)
		do
			echo "$PROG $th $sz $times"
			$PROG $FAST_MEM $SLOW_MEM $sz
		done
	done
done

DONEDATE=$(date +%Y%m%d.%H%M%S)
echo "Done: $DONEDATE"
