#!/usr/bin/env bash

MODE=non-dp
MODE=dp

z_dim=64
num_iters=100
norm=standarize

if [[ ${MODE} = 'non-dp' ]]
then
    # Non-dp
    for beta in 0.001; do #0.01 0.1 1.; do
    python wb_mia.py \
        -name aml_${num_iters}/z${z_dim}_test0.2_beta${beta} \
        -z_dim ${z_dim} \
        --beta ${beta} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --device_type cuda
    done
fi


if [[ ${MODE} = 'dp' ]]
then
    for maxnorm in 0.01; do
    for beta in 0.001; do #0.01 0.1 1.; do
    python wb_mia.py \
        -name dp_aml_${num_iters}/z${z_dim}_test0.2_beta${beta}_dp_norm${maxnorm}_sigma${sigma} \
        -z_dim ${z_dim} \
        --beta ${beta} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --device_type cuda 
    done
    done
fi
