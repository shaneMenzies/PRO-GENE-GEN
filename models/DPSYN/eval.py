import os, sys
import argparse
import pickle
import shutil

sys.path.append("../..")
from utils import *

DATA_DIR = "../../data/"


def parse_arguments():
    parser = argparse.ArgumentParser()
    ### data config
    parser.add_argument("--exp_name", type=str, default="aml")
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
        default="none",
        choices=["standarize", "minmax", "none", "discretize", "svc_selection", "pca"],
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
        "--if_filter_x",
        type=str2bool,
        default=True,
        help="If filter the features using the 1000 landmark genes",
    )

    ### eval setting
    parser.add_argument("--syn_filename", type=str)
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="default",
        choices=["default", "logistic", "svc_l1"],
    )
    parser.add_argument("--eval_frac", type=float, default=0.2, help="eval fraction")
    parser.add_argument("--test_frac", type=float, default=0.2, help="test fraction")
    parser.add_argument(
        "--if_valid",
        type=str2bool,
        default=False,
        help="If further split test dataset to test/valid",
    )

    args = parser.parse_args()
    return args


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## set up save_diri
    save_dir = os.path.join(os.path.dirname(__file__), "results", args.exp_name)
    # save_dir = os.path.join(SAVE_DIR % (args.dataset, args.model_architecture), args.exp_name)
    mkdir(save_dir)

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
    dset_dir = DATA_DIR
    args, save_dir = check_args(parse_arguments())

    ### Load data
    if args.dataset == "aml":
        dset = AML(
            dset_dir,
            preprocess=args.preprocess,
            if_filter_x=args.if_filter_x,
            **args.preprocess_arg,
        )
        dset_ref = AML(dset_dir, preprocess="none", if_filter_x=args.if_filter_x)
        eval_model_type = (
            "svc_l1" if args.eval_model_type == "default" else args.eval_model_type
        )

    x_dim, y_dim = dset.get_dim()
    print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

    ### Set up Evaluation model
    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset_ref.train_test(
            k=1000, test_fraction=args.test_frac
        )
    else:
        test_x, test_y = dset_ref.dset, dset_ref.anno
    print("size of train dataset: %d" % len(dset))
    eval_subset = int(args.eval_frac * len(test_x))
    if args.if_valid:
        valid_x, valid_y = test_x[:eval_subset], test_y[:eval_subset]
        test_x, test_y = test_x[eval_subset:], test_y[eval_subset:]

        print("size of test dataset: %d" % len(test_x))
        print("size of valid dataset: %d" % len(valid_x))
    else:
        valid_x, valid_y = test_x, test_y

    ### Load results
    root = "temp_data/synthesized_records/"
    # root = save_dir
    # paths = sorted(os.listdir(root))
    filename = args.syn_filename

    # for path in paths:
    if filename.startswith(args.dataset):
        ep = filename.split("_")[-1]
        full_path = os.path.abspath(os.path.join(root, filename))
        print(full_path)
        data = pickle.load(open(full_path, "rb"))
        data = data.df
        fake_label = data["label"]
        fake_data = data.drop(columns=["label"])
        inversed_data = dset._inverse_transform(fake_data.to_numpy())

        # save inversed data
        print(os.path.join(save_dir, f"{args.syn_filename}.csv"))
        save_data_csv(
            os.path.join(save_dir, f"{args.syn_filename}.csv"),
            inversed_data,
            fake_label,
            dset.column_names,
        )

        real_acc = Eval(inversed_data, fake_label, model_type=eval_model_type).efficacy(
            test_x, test_y
        )
        fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
            fake_data, fake_label
        )

        print("=" * 100)
        print("ep= " + str(ep))
        print("real acc: ", str(real_acc))
        print("fake acc: ", str(fake_acc))

        write_csv(
            os.path.join(save_dir, "eval.csv"),
            "ep= " + str(ep),
            [real_acc, fake_acc],
            ["real acc", "fake acc"],
        )
    return


if __name__ == "__main__":
    main()
