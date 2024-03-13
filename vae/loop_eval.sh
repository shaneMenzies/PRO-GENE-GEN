

MODE=dp

z_dim=64
num_iters=10000
norm=standarize

if [[ ${MODE} = 'non-dp' ]]
then
    # Non-dp
    for s in 1000 2000; do
    for beta in 0.001; do #0.01 0.1 1.; do
        python main.py \
        -name aml_${num_iters}/z${z_dim}_test0.2_beta${beta} \
        -s ${s} \
        -z_dim ${z_dim} \
        --beta ${beta} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --num_iters ${num_iters} \
        --enable_privacy False \
        --device_type cuda \
        --if_evaluate  True \
        --if_save_data True
    done
    done
fi


if [[ ${MODE} = 'dp' ]]
then
    for s in 1000 2000; do
    for maxnorm in 0.01; do
    for beta in 0.001; do #0.01 0.1 1.; do
    python main.py \
        -name dp_aml_${num_iters}/z${z_dim}_test0.2_beta${beta}_dp_norm${maxnorm}_sigma${sigma} \
        -s ${s} \
        -z_dim ${z_dim} \
        --beta ${beta} \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --num_iters ${num_iters} \
        --enable_privacy True \
        --max_norm ${maxnorm} \
        --device_type cuda \
        --target_epsilon 10\
        --device_type cuda \
        --if_evaluate  True \
        --if_save_data True
    done
    done
    done
fi