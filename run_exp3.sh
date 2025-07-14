#!/bin/bash

run_experiment() {
    echo "Running: $1"
    CUDA_VISIBLE_DEVICES=1 python main.py --config "$1"

    # Wait for a random time between 1 and 120 seconds
    wait_time=$(( (RANDOM % 120) + 1 ))
    echo "Sleeping for $wait_time seconds..."
    sleep $wait_time
}

run_experiment "./experiment/config/prompt_cam/dino/cub/args_Adapter_SelfParam_CrossParam_8.yaml"
run_experiment "./experiment/config/prompt_cam/dino/cub/args_Adapter_SelfParam_CrossParam_9.yaml"
run_experiment "./experiment/config/prompt_cam/dino/cub/args_Adapter_SelfParam_CrossParam_10.yaml"


