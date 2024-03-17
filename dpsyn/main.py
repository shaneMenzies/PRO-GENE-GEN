import logging
import mkl

import sys

sys.path.append("..")

from dpsyn.exp.exp_dpsyn_gum import ExpDPSynGUM
from dpsyn.parameter_parser import parameter_parser


def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(levelname)s:%(asctime)s: - %(name)s - : %(message)s"
    )

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(args):
    # config the logger
    config_logger()
    ExpDPSynGUM(args)


if __name__ == "__main__":
    args = parameter_parser()
    main(args)
