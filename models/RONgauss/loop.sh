#!/usr/bin/env bash

MODE=non-dp
MODE=dp

z_dim=64
norm=standarize
target_epsilon=8

if [[ ${MODE} = 'non-dp' ]]
then
    # Non-dp
    python main.py \
        -name aml/z${z_dim}_test0.2 \
        -z_dim ${z_dim} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --conditional\
        --if_filter_x
fi


if [[ ${MODE} = 'dp' ]]
then
    python main.py \
        -name dp_aml/z${z_dim}_test0.2_epsilon${target_epsilon} \
        -z_dim ${z_dim} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --enable_privacy\
        --conditional\
        --if_filter_x\
        --target_epsilon ${target_epsilon}
fi
