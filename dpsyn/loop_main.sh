epsilon=8
noise_add_method=A3 #A3 -> Weighted Gaussian

# epsilon=500
# noise_add_method=Non

bash preprocess_data.sh
python main.py \
    --dataset_name aml_filter_tr0.2_alpha0.25_k1000 \
    --num_synthesize_records 944\
    --epsilon ${epsilon}\
    --noise_add_method ${noise_add_method}
python inverse_transform_data.py  \
    --filepath temp_data/synthesized_records/aml_filter_tr0.2_alpha0.25_k1000_${noise_add_method}_${epsilon}.0 \
    --preprocess_arg alpha=0.25
python eval.py --syn_filename aml_filter_tr0.2_alpha0.25_k1000_${noise_add_method}_${epsilon}.0