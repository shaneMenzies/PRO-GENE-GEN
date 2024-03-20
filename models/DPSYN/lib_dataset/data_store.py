import pickle
import os

import config as config


class DataStore:
    def __init__(self, args):
        self.args = args

        self.determine_data_path()
        self.generate_folder()

    def determine_data_path(self):
        synthesized_records_name = "_".join(
            (
                self.args["dataset_name"],
                str(self.args["noise_add_method"]),
                str(self.args["epsilon"]),
            )
        )
        marginal_name = "_".join(
            (
                self.args["dataset_name"],
                str(self.args["noise_add_method"]),
                str(self.args["epsilon"]),
            )
        )

        self.synthesized_records_file = (
            config.SYNTHESIZED_RECORDS_PATH + synthesized_records_name
        )
        self.marginal_file = config.MARGINAL_PATH + marginal_name

    def generate_folder(self):
        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)

    def load_processed_data(self):
        return pickle.load(
            open(config.PROCESSED_DATA_PATH + self.args["dataset_name"], "rb")
        )

    def save_synthesized_records(self, records):
        pickle.dump(records, open(self.synthesized_records_file, "wb"))

    def save_marginal(self, marginals):
        pickle.dump(marginals, open(self.marginal_file, "wb"))

    def load_marginal(self):
        return pickle.load(open(self.marginal_file, "rb"))
