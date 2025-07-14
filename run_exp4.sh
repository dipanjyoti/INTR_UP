#!/bin/bash

run_experiment() {
    echo "Running: $1"
    CUDA_VISIBLE_DEVICES=0 python main.py --config "$1"

    # Wait for a random time between 1 and 120 seconds
    wait_time=$(( (RANDOM % 120) + 1 ))
    echo "Sleeping for $wait_time seconds..."
    sleep $wait_time
}

run_experiment "./experiment/config/prompt_cam/dino/gbcu/args_SelfNonParam_CrossNonParam_1.yaml"
run_experiment "./experiment/config/prompt_cam/dino/gbcu/args_SelfNonParam_CrossNonParam_2.yaml"
run_experiment "./experiment/config/prompt_cam/dino/gbcu/args_SelfNonParam_CrossNonParam_3.yaml"
run_experiment "./experiment/config/prompt_cam/dino/gbcu/args_SelfNonParam_CrossNonParam_4.yaml"
run_experiment "./experiment/config/prompt_cam/dino/gbcu/args_SelfNonParam_CrossNonParam_5.yaml"



