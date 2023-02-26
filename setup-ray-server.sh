#!/bin/bash

nodes=`cat $PBS_NODEFILE| sort | uniq`
hostip=`hostname --ip-address`
head_address="$hostip:5711"

n=0
while read -r line; do
    node=$(echo "$line" | awk '{print $NF}')
    node=${node%.*.*.*.*}
    # echo "$node"
    if [ $n -eq 0 ]; 
    then
        echo "Starting head node"
        ray start --head --port=5711 --node-ip-address=$hostip > /dev/null 2>&1 &
        n=1
    else
        echo "Starting worker node"
        ssh $node 
        ray start --address=$head_address
        exit 0
    fi
done <<< "$nodes"

