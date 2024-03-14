#!/usr/bin/env bash

# MODE=non-dp
MODE=dp

z_dim=512
num_epochs=1000
norm=standarize
micro_batch_size=1

if [[ ${MODE} = 'non-dp' ]]
then
    # Non-dp
    python main.py \
        -name aml_${num_epochs}/z${z_dim}_test0.2\
        -z_dim ${z_dim} \
        --test_frac 0.2 \
        --preprocess ${norm}\
        --if_filter_x\
        --num_epochs ${num_epochs}\
        --if_save_model\
        --if_save_data\
        --random_seed 5000\
        --if_verbose
fi


if [[ ${MODE} = 'dp' ]]
then
    python main.py \
        -name dp_aml_${num_epochs}/z${z_dim}_test0.2 \
        -z_dim ${z_dim} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --micro_batch_size ${micro_batch_size}\
        --if_filter_x\
        --num_epochs ${num_epochs}\
        --enable_privacy\
        --target_epsilon 8\
        --if_save_model\
        --if_verbose
fi
