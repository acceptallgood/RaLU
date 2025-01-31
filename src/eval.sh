#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Usage: $0 <file> <dataset>"
    exit 1
fi

file="$1"
dataset="$2"

docker run --rm -v $(pwd)/../$fine:/app ganler/evalplus:latest \
           evalplus.evaluate --dataset "$dataset" \
           --samples "/app/$file" #--pull=always 
           
            