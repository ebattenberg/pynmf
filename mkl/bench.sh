#!/bin/bash

# default vaules
ITER=300
threads_low=1
threads_high=1
TRIALS=20

help_str="usage: bench.sh {send,run} iter[300] trials[20] threads_low[1] threads_high[1]"

if [ -n "$1" ]; then
    if [ "$1" == "-h" ]; then
	echo $help_str
	exit
    elif [ "$1" == "send" ]; then
	scp -r ./results/ zen.millennium.berkeley.edu:/home/eecs/ericb/
	exit
    elif [ "$1" != "run" ]; then
	echo $help_str
	exit
    fi
else
    echo $help_str
    exit
fi
if [ -n "$2" ]; then ITER=$2; echo here; fi
if [ -n "$3" ]; then TRIALS=$3; fi
if [ -n "$4" ]; then threads_low=$4; fi
if [ -n "$5" ]; then threads_high=$5; fi


echo "benchmarking..."
echo "iterations: $ITER"
echo "threads: ${threads_low}-${threads_high}"
echo "trials: $TRIALS"
grep model /proc/cpuinfo | head -n 2
echo "processors: `grep processor /proc/cpuinfo -c`"

pwd
make bench
	
if [ ! -d "./results" ]; then
    mkdir results
fi

rm -f ./results/*.out


for threads in `seq  $threads_low $threads_high`; do
    echo "./bench $ITER $TRIALS $threads $threads"
    ./bench $ITER $TRIALS $threads $threads
done

echo $HOSTNAME

if [ -d /home/eecs/ericb/results ]; then
    if [ ! -d /home/eecs/ericb/results/$HOSTNAME ]; then
	mkdir -p /home/eecs/ericb/results/$HOSTNAME
    fi

    num=`date +%s`
    mkdir /home/eecs/ericb/results/$HOSTNAME/$num

    mv ./results/* /home/eecs/ericb/results/$HOSTNAME/$num/

    rmdir ./results
else
    num=`date +%s`
    mkdir -p ./results/$HOSTNAME/$num
    mv ./results/*.out ./results/$HOSTNAME/$num
    #scp -r ./results/ zen.millennium.berkeley.edu:/home/eecs/ericb/
fi
