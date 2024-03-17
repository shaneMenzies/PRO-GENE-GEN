import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parameter_parser():
    parser = argparse.ArgumentParser()

    ####################################### general parameters ###################################
    # parameters for single run
    parser.add_argument("--dataset_name", type=str, default="aml", help="options: aml")

    # parameters for workflow control
    parser.add_argument("--is_cal_marginals", type=str2bool, default=True)
    parser.add_argument("--is_cal_depend", type=str2bool, default=True)

    # parameters for privacy
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=8.0,
        help="when run main(), specify epsilon here",
    )
    parser.add_argument("--depend_epsilon_ratio", type=float, default=0.1)
    parser.add_argument("--marg_add_sensitivity", type=float, default=1.0)
    parser.add_argument("--marg_select_sensitivity", type=float, default=4.0)
    parser.add_argument(
        "--noise_add_method",
        type=str,
        default="A3",
        help="A1 -> Equal Laplace; A2 -> Equal Gaussian; A3 -> Weighted Gaussian",
    )

    # parameters for marginal selection
    parser.add_argument("--is_combine", type=str2bool, default=True)

    ############################################# specific parameters ############################################
    # parameters for view consist and non-negativity
    parser.add_argument(
        "--non_negativity",
        type=str,
        default="N3",
        help="N1 -> norm_cut; N2 -> norm_sub; N3 -> norm_sub + norm_cut",
    )
    parser.add_argument("--consist_iterations", type=int, default=501)

    # parameters for synthesizing
    parser.add_argument("--initialize_method", type=str, default="singleton")
    parser.add_argument(
        "--update_method",
        type=str,
        default="S5",
        help="S1 -> all replace; S2 -> all duplicate; S3 -> all half-half;"
        "S4 -> replace+duplicate; S5 -> half-half+duplicate; S6 -> half-half+replace."
        "The optimal one is S5",
    )
    parser.add_argument("--append", type=str2bool, default=True)
    parser.add_argument("--sep_syn", type=str2bool, default=False)

    parser.add_argument(
        "--update_rate_method",
        type=str,
        default="U4",
        help="U4 -> step decay; U5 -> exponential decay; U6 -> linear decay; U7 -> square root decay."
        "The optimal one is U4",
    )
    parser.add_argument("--update_rate_initial", type=float, default=1.0)
    parser.add_argument("--num_synthesize_records", type=int, default=int(6e5))
    parser.add_argument("--update_iterations", type=int, default=200)

    return vars(parser.parse_args())
