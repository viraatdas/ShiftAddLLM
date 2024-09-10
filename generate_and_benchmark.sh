#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model> <load>"
    exit 1
fi

# Assign arguments to variables
MODEL=$1
LOAD=$2

# Set CUDA devices and run the Python script with passed arguments
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model "$MODEL" --load "$LOAD" --benchmark
