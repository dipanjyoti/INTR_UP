#!/bin/bash

run_experiment() {
    echo "Running: $1"
    CUDA_VISIBLE_DEVICES=3 python main.py --config "$1"

    # Wait for a random time between 1 and 60 seconds
    wait_time=$(( (RANDOM % 120) + 1 ))
    echo "Sleeping for $wait_time seconds..."
    sleep $wait_time
}

run_experiment "./experiment/config/prompt_cam/bioclip2/cub/args_SelfParam_CrossParam_1.yaml"
run_experiment "./experiment/config/prompt_cam/bioclip2/cub/args_SelfParam_CrossParam_2.yaml"


