import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import pickle
import shutil
import collections

sys.path.append("../..")
from utils import *

from model import DP_WGAN, Generator, Discriminator

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
        "--random_seed", "-s", type=int, default=1000, help="random seed"
    )
    parser.add_argument(
        "--split_seed", type=int, default=1000, help="random seed for data splitting"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--train_batchsize", type=int, default=64, help="training batch size"
    )
    parser.add_argument(
        "--num_epochs", "-ep", type=int, default=1000, help="number of epochs"
    )
    parser.add_argument("--z_dim", "-z_dim", type=int, default=512, help="z_dim")
    parser.add_argument("--test_frac", type=float, default=0, help="test fraction")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="standarize",
        choices=["standarize", "minmax", "none", "discretize"],
        help="pre-processing scheme",
    )
    parser.add_argument(
        "--num_samples_ratio",
        type=int,
        default=1,
        help="num of generated samples(ratio to ori size)",
    )
    parser.add_argument(
        "--if_filter_x",
        action="store_true",
        default=False,
        help="If filter the features",
    )
    parser.add_argument(
        "--if_uniform_y",
        action="store_true",
        default=False,
        help="If sample y uniformly",
    )
    parser.add_argument(
        "--if_save_model",
        "-save",
        action="store_true",
        default=False,
        help="If save the final model",
    )
    parser.add_argument(
        "--if_resume",
        action="store_true",
        default=False,
        help="If restore the training",
    )
    parser.add_argument(
        "--if_evaluate",
        action="store_true",
        default=False,
        help="If only eval the model",
    )
    parser.add_argument(
        "--if_save_data",
        action="store_true",
        default=False,
        help="If save the fake data",
    )
    parser.add_argument(
        "--if_verbose",
        action="store_true",
        default=False,
        help="To evaluate during training",
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
    # noisy_sgd_parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--sigma",
        type=float,
        default=2,
        help="Gaussian noise variance multiplier. A larger sigma will make the model "
        "train for longer epochs for the same privacy budget",
    )
    parser.add_argument(
        "--clip_coeff",
        type=float,
        default=0.1,
        help="The coefficient to clip the gradients before adding noise for private "
        "SGD training",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Parameter to tradeoff speed vs efficiency. Gradients are averaged for a microbatch "
        "and then clipped before adding noise",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    # parser_dp_wgan = subparsers.add_parser('dp-wgan', parents=[privacy_parser, noisy_sgd_parser])
    parser.add_argument(
        "--if_onehot",
        action="store_true",
        default=False,
        help="If using onehot encoding for class variable y",
    )
    parser.add_argument(
        "--clamp_lower",
        type=float,
        default=-0.01,
        help="Clamp parameter for wasserstein GAN",
    )
    parser.add_argument(
        "--clamp_upper",
        type=float,
        default=0.01,
        help="Clamp parameter for wasserstein GAN",
    )
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
    if not args.if_evaluate:
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

    ### CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    ### Random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)

    ### Prepare Data
    dset_dir = DATA_DIR
    save_path = os.path.join(save_dir, "checkpoint.pkt")
    dset = AML(dset_dir, preprocess=args.preprocess, if_filter_x=args.if_filter_x)
    eval_model_type = "svc_l1"
    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset.train_test(
            k=args.split_seed, test_fraction=args.test_frac
        )
        eval_model = Eval(test_x, test_y)
    else:
        train_x, train_y = dset.dset, dset.anno
        eval_model = Eval(dset.dset, dset.anno)
    labels, class_ratios = np.unique(train_y, return_counts=True)
    class_ratios = class_ratios / len(train_y)
    x_dim, y_dim = dset.get_dim()
    data_loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator(device="cuda"),
    )
    assert len(labels) == y_dim
    print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

    ### Set up model
    input_dim = x_dim
    z_dim = args.z_dim
    conditional = True
    onehot = args.if_onehot
    dset_size = len(train_x)
    target_epsilon = args.target_epsilon
    target_delta = args.target_delta
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    lr = 5e-5
    clamp_upper = args.clamp_upper
    clamp_lower = args.clamp_lower
    clip_coeff = args.clip_coeff
    sigma = args.sigma
    num_epochs = args.num_epochs
    Hyperparams = collections.namedtuple(
        "Hyperarams",
        "batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs",
    )
    Hyperparams.__new__.__defaults__ = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    hyper = Hyperparams(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        clamp_lower=clamp_lower,
        clamp_upper=clamp_upper,
        clip_coeff=clip_coeff,
        sigma=sigma,
        class_ratios=class_ratios,
        lr=lr,
        num_epochs=num_epochs,
    )

    if conditional:
        if onehot:
            z = z_dim + y_dim
        else:
            z = z_dim + 1
        d = input_dim
    else:
        z = z_dim
        d = input_dim + 1
    generator = Generator(z, d).cuda().double()
    if onehot:
        discriminator = (
            Discriminator(input_dim + y_dim, wasserstein=True).cuda().double()
        )
    else:
        discriminator = Discriminator(input_dim + 1, wasserstein=True).cuda().double()
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    model = DP_WGAN(
        input_dim,
        z_dim,
        y_dim,
        dset_size,
        data_loader,
        generator,
        discriminator,
        target_epsilon,
        target_delta,
        hyperparams=hyper,
        conditional=conditional,
        onehot=onehot,
    )

    ### set up logger
    title = "aml_dpwgan"
    logger = Logger(os.path.join(save_dir, "log.txt"), title=title)
    logger.set_names(["epsilon", "Real Acc", "Fake Acc"])

    ### Train
    epsilon = 0
    gen_iters = 0
    steps = 0
    epoch = 0
    real_acc = 0
    fake_acc = 0
    Stats = collections.namedtuple("Stats", "epsilon gen_iters steps epoch")
    Stats.__new__.__defaults__ = (None, None, None, None)

    if args.if_evaluate:
        checkpoint = torch.load(save_path)  # load the checkpoint
        epoch_start = checkpoint["epoch"] + 1
        generator.load_state_dict(checkpoint["gen_state_dict"])
        discriminator.load_state_dict(checkpoint["disc_state_dict"])

        fake_data = []
        fake_label = []
        if args.if_uniform_y:
            class_ratios = (
                np.ones(
                    y_dim,
                )
                / y_dim
            )
        for i in range(args.num_samples_ratio):
            syn_data = model.generate(train_x.shape[0], class_ratios)
            X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]
            fake_data.append(X_syn)
            fake_label.append(y_syn)
            fake_data_ = np.concatenate(fake_data)
            fake_label_ = np.concatenate(fake_label)

            if args.if_evaluate:
                real_acc = Eval(
                    fake_data_, fake_label_, model_type=eval_model_type
                ).efficacy(test_x, test_y)
                fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
                    fake_data_, fake_label_
                )
                print("=" * 100)
                print("real acc: ", str(real_acc))
                print("fake acc: ", str(fake_acc))
                write_csv(
                    os.path.join(save_dir, "eval.csv"),
                    "ratio= " + str(i) + " uniform" if args.if_uniform_y else "",
                    [real_acc, fake_acc],
                    ["real acc", "fake acc"],
                )

        if args.if_save_data:
            save_data_csv(
                os.path.join(
                    save_dir, f"samples/k{args.split_seed}_s{args.random_seed}.csv"
                ),
                dset._inverse_transform(fake_data_),
                fake_label_,
                dset.column_names,
            )

    else:
        while epsilon < target_epsilon:
            curr_stats = Stats(
                epsilon=epsilon, gen_iters=gen_iters, steps=steps, epoch=epoch
            )
            epsilon, epoch, steps, gen_iters = model.train(
                curr_stats, private=args.enable_privacy, conditional=True
            )

            if epoch % 50 == 0:
                ### eval
                syn_data = model.generate(train_x.shape[0], class_ratios)
                fake_data, fake_label = syn_data[:, :-1], syn_data[:, -1]
                real_acc = Eval(
                    fake_data, fake_label, model_type=eval_model_type
                ).efficacy(test_x, test_y)
                fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
                    fake_data, fake_label
                )

            if args.if_save_model:
                save_dict = {
                    "epoch": epoch,
                    "gen_state_dict": generator.state_dict(),
                    "disc_state_dict": discriminator.state_dict(),
                }
                torch.save(save_dict, save_path)

            logger.append([epsilon, real_acc, fake_acc])
            logger.plot(["Real Acc", "Fake Acc"])
            savefig(os.path.join(save_dir, "acc.png"))

        if args.if_verbose:
            ### Eval
            fake_data = []
            fake_label = []
            if args.if_uniform_y:
                class_ratios = (
                    np.ones(
                        y_dim,
                    )
                    / y_dim
                )
            for i in range(args.num_samples_ratio):
                syn_data = model.generate(train_x.shape[0], class_ratios)
                X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]
                fake_data.append(X_syn)
                fake_label.append(y_syn)
                fake_data_ = np.concatenate(fake_data)
                fake_label_ = np.concatenate(fake_label)
                real_acc = Eval(
                    fake_data_, fake_label_, model_type=eval_model_type
                ).efficacy(test_x, test_y)
                fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
                    fake_data_, fake_label_
                )
                print("=" * 100)
                print("real acc: ", str(real_acc))
                print("fake acc: ", str(fake_acc))
                write_csv(
                    os.path.join(save_dir, "eval.csv"),
                    "ratio= " + str(i) + " uniform" if args.if_uniform_y else "",
                    [real_acc, fake_acc],
                    ["real acc", "fake acc"],
                )
            if args.if_save_data:
                save_data_csv(
                    os.path.join(
                        save_dir, f"samples/k{args.split_seed}_s{args.random_seed}.csv"
                    ),
                    dset._inverse_transform(fake_data_),
                    fake_label_,
                    dset.column_names,
                )
    return


if __name__ == "__main__":
    main()
