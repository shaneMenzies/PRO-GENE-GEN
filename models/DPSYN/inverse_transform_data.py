import os, sys
import argparse
import pickle

sys.path.append("../..")
from utils import *

DATA_DIR = "../../data/"


def parse_arguments():
    parser = argparse.ArgumentParser()
    ### data config
    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="aml",
        choices=["aml"],
        help="dataset name",
    )
    parser.add_argument("--filepath", type=str, help="data file path")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="discretize",
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


def main():
    ### config
    args = parse_arguments()
    dset_dir = DATA_DIR
    filename = os.path.basename(args.filepath)
    save_dir = os.path.dirname(args.filepath)
    mkdir(save_dir)

    ### Arguments (shown as the filename)
    k = 1000
    preprocess = args.preprocess
    print(args.preprocess_arg)
    preprocess_args = ""
    for key, value in args.preprocess_arg.items():
        preprocess_args += "_" + str(key) + str(value)
    print(preprocess_args)
    filter_x = "filterx" if args.if_filter_x else "fullx"

    ### Load data
    if args.dataset == "aml":
        dset = AML(
            dset_dir,
            preprocess=args.preprocess,
            if_filter_x=args.if_filter_x,
            **args.preprocess_arg,
        )

    x_dim, y_dim = dset.get_dim()

    ### Set up Evaluation model
    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset.train_test(
            k=k, test_fraction=args.test_frac
        )
    else:
        test_x, test_y = dset.dset, dset.anno
    print("size of train dataset: %d" % len(dset))
    eval_subset = int(args.eval_frac * len(test_x))
    if args.if_valid:
        valid_x, valid_y = test_x[:eval_subset], test_y[:eval_subset]
        test_x, test_y = test_x[eval_subset:], test_y[eval_subset:]

        print("size of test dataset: %d" % len(test_x))
        print("size of valid dataset: %d" % len(valid_x))
    else:
        valid_x, valid_y = test_x, test_y

    ## read (synthetic data file)
    # with open(args.filepath, 'rb') as f:
    #     result = chardet.detect(f.read())
    # data_df = pd.read_csv(args.filepath, header=0, encoding=result['encoding'])
    data = pickle.load(open(args.filepath, "rb"))
    data_df = data.df
    save_data_csv(
        os.path.join(save_dir, f"{filename}_inverse.csv"),
        dset._inverse_transform(data_df),
        data_df["label"],
        data_df.columns[1:],  # avoid duplicate 'label' entry
    )


if __name__ == "__main__":
    main()
