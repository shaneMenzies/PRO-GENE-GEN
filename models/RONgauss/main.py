import os, sys
import argparse
import pickle
import shutil

sys.path.append("../..")
from model import RONGauss
from utils import *

DATA_DIR = "../../data/"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        "-name",
        type=str,
        required=True,
        help="path for storing the checkpoint",
    )
    parser.add_argument(
        "--split_seed", type=int, default=1000, help="random seed for data splitting"
    )
    parser.add_argument(
        "--random_seed", "-s", type=int, default=1000, help="random seed"
    )

    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="aml",
        choices=["aml", "mouse"],
        help="dataset name",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="standarize",
        choices=["standarize", "minmax", "none", "discretize"],
        help="pre-processing scheme",
    )
    parser.add_argument(
        "--preprocess_arg",
        metavar="KEY=VALUE",
        default={},
        nargs="+",
        action=ParseDict,
        help="set the arguments for preprocess using key-value pairs",
    )

    parser.add_argument(
        "--num_samples_ratio",
        type=int,
        default=1,
        help="num of generated samples(ratio to ori size)",
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="default",
        choices=["default", "logistic", "svc_l1"],
    )
    parser.add_argument("--eval_frac", type=float, default=0.2, help="eval fraction")
    parser.add_argument("--test_frac", type=float, default=0.2, help="test fraction")

    parser.add_argument(
        "--if_filter_x",
        action="store_true",
        default=False,
        help="If filter the features",
    )

    # subparsers = parser.add_subparsers(help="generative model type", dest="model")
    # privacy_parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable_privacy", action="store_true", help="Enable private data generation"
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=8,
        help="Epsilon differential privacy parameter",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="Delta differential privacy parameter",
    )
    # model_parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--z_dim", "-z_dim", type=int, default=64, help="z_dim")
    parser.add_argument(
        "--conditional",
        action="store_true",
        default=False,
        help="if call the conditional generation",
    )

    # parser_ron_gauss = subparsers.add_parser('ron-gauss', parents=[privacy_parser, model_parser])
    args = parser.parse_args()
    return args


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), "results", args.exp_name)
    # save_dir = os.path.join(SAVE_DIR % (args.dataset, args.model_architecture), args.exp_name)
    mkdir(save_dir)
    mkdir(save_dir + "/samples")

    ## store the parameters
    with open(os.path.join(save_dir, "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(
        vars(args), open(os.path.join(save_dir, "params.pkl"), "wb"), protocol=2
    )

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Load data
    dset_dir = DATA_DIR
    if args.dataset == "aml":
        dset = AML(
            dset_dir,
            preprocess=args.preprocess,
            if_filter_x=args.if_filter_x,
            **args.preprocess_arg,
        )
        eval_model_type = (
            "svc_l1" if args.eval_model_type == "default" else args.eval_model_type
        )

    x_dim, y_dim = dset.get_dim()
    print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset.train_test(
            k=args.split_seed, test_fraction=args.test_frac
        )
    else:
        train_x, train_y = dset.dset, dset.anno
        test_x, test_y = dset.dset, dset.anno
    x_dim, y_dim = dset.get_dim()
    print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

    ### Set up model and training
    model = RONGauss(
        args.z_dim,
        args.target_epsilon,
        args.target_delta,
        args.conditional,
        args.enable_privacy,
    )
    X_syn, y_syn, dp_mean = model.generate(
        train_x,
        y=train_y,
        centering=False,
        max_y=4,
        prng_seed=args.random_seed,
        if_uniform_y=False,
    )

    ### Eval
    real_acc = Eval(X_syn, y_syn, model_type=eval_model_type).efficacy(test_x, test_y)
    fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(X_syn, y_syn)
    print("=" * 100)
    print("real acc: ", str(real_acc))
    print("fake acc: ", str(fake_acc))

    save_data_csv(
        os.path.join(save_dir, f"samples/k{args.split_seed}_s{args.random_seed}.csv"),
        dset._inverse_transform(X_syn),
        y_syn,
        dset.column_names,
    )

    write_csv(
        os.path.join(save_dir, "eval.csv"),
        "ratio= 1",
        [real_acc, fake_acc],
        ["real acc", "fake acc"],
    )
    return


if __name__ == "__main__":
    main()
