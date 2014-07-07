#!/bin/bash

# default vaules
ITER=200
block_size_low=128
block_size_high=128
TRIALS=20

help_str="usage: bench.sh {run,send} iter[300] trials[20] block_size_low[128] block_size_high[128]"

if [ -n "$1" ]; then
    if [ "$1" == "-h" ]; then
	echo $help_str
	exit
    elif [ "$1" == "send" ]; then
	scp -r ./results/ zen.millennium.berkeley.edu:/home/eecs/ericb/
	exit
    fi
else
    echo $help_str
    exit
fi
if [ -n "$2" ]; then ITER=$2; echo here; fi
if [ -n "$3" ]; then TRIALS=$3; fi
if [ -n "$4" ]; then block_size_low=$4; fi
if [ -n "$5" ]; then block_size_high=$5; fi


echo "benchmarking..."
echo "iterations: $ITER"
echo "block_size: ${threads_low}-${threads_high}"
echo "trials: $TRIALS"

pwd
make bench
	
if [ ! -d ./results ]; then
    mkdir results
fi

rm -f ./results/*.out

block_size=$block_size_low
BLOCK_SIZES=$block_size
while [ $[$block_size*2] -le $block_size_high ]; do
    block_size=$[$block_size*2]
    BLOCK_SIZES="$BLOCK_SIZES $block_size"
done
echo $BLOCK_SIZES

for block_size in $BLOCK_SIZES; do
    echo "./bench $ITER $TRIALS $block_size $block_size"
    ./bench $ITER $TRIALS $block_size $block_size
done

HOSTNAME=${HOSTNAME}_d 
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
