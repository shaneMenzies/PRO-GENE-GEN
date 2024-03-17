import os, sys
import logging
import pickle
import json
import ssl
import zipfile
from six.moves import urllib
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

sys.path.append("..")
import dpsyn.config as config
from dpsyn.lib_dataset.dataset import Dataset
from dpsyn.lib_dataset.domain import Domain

DATA_DIR = "../data/aml"


class PreprocessAML:
    def __init__(self):
        self.logger = logging.getLogger("preprocess AML")
        self.shape = []
        self.gene_path = None

        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)

    def select_features(self):
        """
        filter the most important genes
        :return:
        """
        self.gene_path = os.path.join(DATA_DIR, "L1000_landmark_gene_list.txt")
        print("Select the 1000 most important genes")

    def load_data(self):
        self.logger.info("loading data")
        self.df = pd.read_csv(
            os.path.join(DATA_DIR, "norm_counts_AML.txt"),
            sep="\t",
            header=0,
            index_col=0,
        )
        self.df = self.df.T
        self.anno_path = os.path.join(DATA_DIR, "annotation_AML.txt")

        ### load labels
        self.classes = ["ALL", "AML", "CLL", "CML"]
        self.anno = []
        with open(self.anno_path) as f:
            f.readline()  # Discard the header manually.
            # Each line consists of Dataset, GSE, Condition, Disease, Tissue, FAB, and Filename
            for line in f.readlines():
                line = line.strip()
                label = line.split("\t")[-4]  # Get the Disease label
                if label not in self.classes:  # convert to 5 classes setting
                    label = "Other"
                self.anno.append(label)
        ### encode labels
        label_encoder = LabelEncoder()
        self.anno = label_encoder.fit_transform(self.anno)
        self.label_map = label_encoder.classes_
        self.label_dict = {}
        for ind, ll in enumerate(self.label_map):
            self.label_dict[ll] = ind

        ### select important genes
        if self.gene_path is not None:
            gene = []
            with open(self.gene_path) as f:
                f.readline()  # Discard the header manually.
                for line in f.readlines():
                    line = line.strip()
                    gene.append(line)
            gene = set(gene)

            extract_dset = []
            column_names = []
            for g in self.df.columns:
                if g in gene:
                    column_names.append(g)
                    extract_dset.append(self.df[g])
            extract_dset = np.array(extract_dset).T
            self.df = pd.DataFrame(extract_dset, columns=column_names)

    def discretize(self, alpha=0.2):
        """
        discretize the numerical attributes
        :param alpha:
        :return:
        """

        assert (alpha < 0.5, "the alpha (quantile) should be smaller than 0.5")
        alphas = [
            alpha,
            0.5,
            1 - alpha,
        ]  # quantiles for the discretization (let num_active = num_inactive)
        bin_number = len(alphas) + 1
        data_quantile = np.quantile(self.df, alphas, axis=0)
        column_names = self.df.columns

        # discretization given the pre-defined
        data_discrete = []
        statistic_arr = []  # storing the discritization results
        mean_arr = []  # storing the mean for each bin
        for idx, col in enumerate(self.df.columns):
            discrete_col = np.digitize(self.df[col], data_quantile[:, idx])
            data_discrete.append(discrete_col)

            # store the results (for inverse_transform)
            statistic_arr.append([])
            mean_arr.append([])
            for bin_idx in range(bin_number):
                curr_col = self.df[col]
                bin_arr = curr_col[discrete_col == bin_idx]
                statistic_arr[idx].append(bin_arr)
                mean_arr[idx].append(np.mean(bin_arr))
        data_discrete = np.array(data_discrete).T
        self.df = pd.DataFrame(data_discrete, columns=column_names)
        self.statistic_arr = statistic_arr
        self.mean_arr = np.array(mean_arr)
        self.bin_number = bin_number

    def inverse_discretize(self, df_in):
        data_processed = []
        for idx, col in enumerate(self.df.columns):
            mean_labels = self.mean_arr[idx]
            data_processed.append(mean_labels[df_in[col]])
        data_processed = np.array(data_processed).T
        return data_processed

    def merge(self):
        """
        merge the feature with the label annotation
        :return:
        """
        merged = np.hstack((np.expand_dims(self.anno, -1), self.df))
        column_names = self.df.columns
        self.df = pd.DataFrame(
            merged, columns=np.concatenate((["label"], column_names))
        )

    def save_data(self, name=None):
        self.logger.info("saving data")

        domain = Domain(self.df.columns, self.shape)
        dataset = Dataset(self.df, domain)

        if name is not None:
            pickle.dump(dataset, open(config.PROCESSED_DATA_PATH + name, "wb"))
        else:
            pickle.dump(dataset, open(config.PROCESSED_DATA_PATH + "aml", "wb"))

        self.logger.info("saved data")

    def calculate_num_categories(self):
        self.logger.info("calculating num_categories")

        for index, col in enumerate(self.df.columns):
            maxval = np.max(self.df[col])
            self.shape.append(maxval + 1)

        self.logger.info("calculated num_categories")

    def train_test(self, k=1000, test_fraction=0.2):
        Train, Test = train_test_split(self.df, test_size=test_fraction, random_state=k)
        self.df = Train
        print(Train.head())
        print(len(Train), len(Test))
        return Train, Test


def main():
    # os.chdir("../")

    output_file = None
    k = 1000
    alpha = 0.25
    logging.basicConfig(
        filename=output_file,
        format="%(levelname)s:%(asctime)s: - %(name)s - : %(message)s",
        level=logging.DEBUG,
    )

    preprocess = PreprocessAML()
    preprocess.select_features()
    preprocess.load_data()
    preprocess.discretize(alpha=alpha)
    preprocess.merge()
    preprocess.train_test(k=k, test_fraction=0.2)
    preprocess.calculate_num_categories()
    preprocess.save_data(f"aml_filter_tr0.2_alpha{alpha}_k{k}")


if __name__ == "__main__":
    main()
