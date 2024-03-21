# Evaluation of Biology preserved in Synthetic Data

In order to reproduce the bio-evaluation results, procede as follows:

1. **Download the synthetic data pre-processed for the `hcocena` pipeline.**

Preprocessing here ony refers to reshaping, no other changes have been performed. Reshaping was necessary because `hcocena` requires as input count data in the shape *genes x samples* and a separate annotation table stating the label per sample.

The data can be downloaded [here](https://dl.cispa.de/s/fgL5StLMpKoa6CE/download/eval_data.zip).

Please move the contents of the zipped file to `/bio_eval/data/`.


2. **Run `hcocena` analysis.**

For this, please use `hcocena_main.RMD`.


## Figure reconstruction

The hcocena results for all models and privacy budgets have been summarised and can be found in `outputs/`. The figures can be reconstructed with these results using `plot-DE-TPR-FPR.R` (FPR/TPR plot) and `plot_coex.R` (coexpression plot).
