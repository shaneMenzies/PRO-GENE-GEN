import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import random
import pickle
import shutil
from sklearn import metrics

sys.path.append("../../")
from model import CVAE
from utils import *

DATA_DIR = "../../data/"


def plot_roc(pos_results, neg_results):
    labels = np.concatenate(
        (np.zeros((len(neg_results),)), np.ones((len(pos_results),)))
    )
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    return fpr, tpr, threshold, auc, ap


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
        "--random_seed", "-s", type=int, default=1000, help="random seed"
    )
    parser.add_argument(
        "--split_seed", type=int, default=1000, help="random seed for data splitting"
    )
    parser.add_argument(
        "--device_type", "-d", type=str, default="cuda", help="type of device"
    )

    ### data config
    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="aml",
        choices=["aml"],
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
        "--if_filter_x",
        type=str2bool,
        default=True,
        help="If filter the features using the 1000 landmark genes",
    )

    ### model training config
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="weight for kl loss")
    parser.add_argument(
        "--num_iters", "-iters", type=int, default=10000, help="number of iters"
    )
    parser.add_argument("--z_dim", "-z_dim", type=int, default=64, help="z_dim")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["exp", "none", "tahn", "sigmoid", "relu"],
        help="output transform function",
    )

    ### eval setting
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="default",
        choices=["default", "logistic", "svc_l1"],
    )
    parser.add_argument("--test_frac", type=float, default=0.2, help="test fraction")

    args = parser.parse_args()
    return args


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__), "results", args.dataset, args.exp_name
    )
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
    args, save_dir = check_args(parse_arguments())

    ### Random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ### set device
    device = torch.device(args.device_type)

    ### CUDA
    if device.type == "cuda":
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed_all(args.random_seed)
    else:
        use_mps = torch.backends.mps.is_available()

    ### Paths
    dset_dir = DATA_DIR
    save_path = os.path.join(save_dir, "model.pkt")

    ### Load data
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

    ### Set up Evaluation model
    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset.train_test(
            k=args.split_seed, test_fraction=args.test_frac
        )
    else:
        test_x, test_y = dset.dset, dset.anno
    print("size of train dataset: %d" % len(dset))
    print("size of test dataset: %d" % len(test_x))

    ref_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
        train_x, train_y, seed=args.random_seed
    )
    print("=" * 100)
    print(
        f"reference accuracy={ref_acc}, (classification on real data with dim={x_dim})"
    )

    ### Specify parameters for DP dataloader
    train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    X_tensor = torch.tensor(test_x, dtype=torch.float32)
    y_tensor = torch.tensor(test_y, dtype=torch.int64)
    testset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    ### Set up VAE models and optimizer
    model = CVAE(
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=args.z_dim,
        beta=args.beta,
        transform=args.transform,
    ).to(device)

    try:
        model.load_state_dict(torch.load(save_path))  # load model
    except:
        print("unable to load model")
        pass

    ### eval
    MSE = nn.MSELoss(reduction="none")
    member_losses = []
    for data_x, data_y in train_loader:
        data_x = data_x.to(device)
        data_y = one_hot_embedding(data_y, num_classes=y_dim, device=device)
        mu, logvar, rec = model(data_x, data_y)
        rec_loss = torch.mean(MSE(rec, data_x), dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        # losses = rec_loss + args.beta * kl_loss
        losses = rec_loss
        member_losses.append(losses.detach())
    member_losses = torch.cat(member_losses, dim=0).cpu().numpy()

    nonmember_losses = []
    for data_x, data_y in test_loader:
        data_x = data_x.to(device)
        data_y = one_hot_embedding(data_y, num_classes=y_dim, device=device)
        mu, logvar, rec = model(data_x, data_y)
        rec_loss = torch.mean(MSE(rec, data_x), dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        # losses = rec_loss + args.beta * kl_loss
        losses = rec_loss
        nonmember_losses.append(losses.detach())
    nonmember_losses = torch.cat(nonmember_losses, dim=0).cpu().numpy()

    print("-------MIA-----------")
    fpr, tpr, threshold, auc, ap = plot_roc(-member_losses, -nonmember_losses)
    print(member_losses.shape)
    print(nonmember_losses.shape)
    print(np.min(member_losses), np.max(member_losses), np.mean(member_losses))
    print(np.min(nonmember_losses), np.max(nonmember_losses), np.mean(nonmember_losses))
    print("auc", auc)
    print("ap", ap)
    return


if __name__ == "__main__":
    main()
