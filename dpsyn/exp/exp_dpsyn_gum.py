import datetime
import logging
import math

import numpy as np
import pandas as pd

from dpsyn.exp.exp_dpsyn import ExpDPSyn
from dpsyn.lib_dpsyn.sep_graph import SepGraph
from dpsyn.lib_dpsyn.update_config import UpdateConfig
from dpsyn.lib_dataset.dataset import Dataset
from dpsyn.lib_dpsyn.attr_append import AttrAppend


class ExpDPSynGUM(ExpDPSyn):
    def __init__(self, args):
        super(ExpDPSynGUM, self).__init__(args)

        self.logger = logging.getLogger("exp_dpsyn_gum")

        ################################## main procedure ##########################################
        self.preprocessing()

        self.construct_views()
        self.anonymize_views()
        self.consist_views(self.attr_recode.dataset_recode.domain, self.views_dict)

        self.synthesize_records()
        self.postprocessing()

    def preprocessing(self):
        self.sep_graph = SepGraph(self.original_dataset.domain, self.marginals)
        self.sep_graph.cut_graph()

        self.attr_append = AttrAppend(self.attr_recode.dataset.domain, self.marginals)
        iterate_marginals = self.attr_append.clip_graph(enable=self.args["append"])

        self.iterate_keys = self.sep_graph.find_sep_graph(
            iterate_marginals, enable=self.args["sep_syn"]
        )

    def construct_views(self):
        self.logger.info("constructing views")

        for i, marginal in enumerate(self.marginals):
            self.logger.debug("%s th marginal" % (i,))
            self.views_dict[marginal] = self.construct_view(
                self.attr_recode.dataset_recode, marginal
            )

        # this part can be obtained directly from attrs recode part
        for singleton in self.original_dataset.domain.attrs:
            self.views_dict[(singleton,)] = self.construct_view(
                self.attr_recode.dataset_recode, (singleton,)
            )
            self.singleton_key.append((singleton,))

    def anonymize_views(self):
        self.logger.info("anonymizing views")

        divider = 0.0

        for key, view in self.views_dict.items():
            divider += math.sqrt(view.num_key)

        for key, view in self.views_dict.items():
            if self.args["noise_add_method"] == "A1":
                view.rho = 1.0
                self.anonymize_view(
                    view, epsilon=self.remain_epsilon / len(self.views_dict)
                )
            elif self.args["noise_add_method"] == "A2":
                view.rho = self.remain_rho / len(self.views_dict)
                self.anonymize_view(view, rho=view.rho)
            elif self.args["noise_add_method"] == "A3":
                view.rho = self.remain_rho * math.sqrt(view.num_key) / divider
                self.anonymize_view(view, rho=view.rho)
            elif self.args["noise_add_method"] == "Non":
                # view.rho = self.remain_rho * math.sqrt(view.num_key) /
                view.rho = 1.0
                pass
            else:
                raise Exception("invalid noise adding method")

    def synthesize_records(self):
        self.synthesized_df = pd.DataFrame(
            data=np.zeros(
                [self.args["num_synthesize_records"], self.num_attributes],
                dtype=np.uint32,
            ),
            columns=self.original_dataset.domain.attrs,
        )
        self.error_tracker = pd.DataFrame()

        # main procedure for synthesizing records
        for key, value in self.iterate_keys.items():
            self.logger.info("synthesizing for %s" % (key,))

            synthesizer = self._update_records(value)
            self.synthesized_df.loc[:, key] = synthesizer.update.df.loc[:, key]
            print(self.error_tracker)
            print(synthesizer.update.error_tracker)

            # self.error_tracker = self.error_tracker.append(synthesizer.update.error_tracker)
            # note: append is now deprecated in pandas 2.0

            self.error_tracker = pd.concat(
                [self.error_tracker, synthesizer.update.error_tracker]
            )

    def postprocessing(self):
        self.logger.info("postprocessing dataset")

        # decode records
        self.sep_graph.join_records(self.synthesized_df)
        self.attr_append.append_attrs(self.synthesized_df, self.views_dict)
        self.attr_recode.decode(self.synthesized_df)

        self.synthesized_dataset = Dataset(
            self.synthesized_df, self.original_dataset.domain
        )
        self.end_time = datetime.datetime.now()

        self.data_store.save_synthesized_records(self.synthesized_dataset)

    def _update_records(self, views_iterate_key):
        update_config = {
            "alpha": self.args["update_rate_initial"],
            "alpha_update_method": self.args["update_rate_method"],
            "update_method": self.args["update_method"],
            "threshold": 0.0,
        }

        singletons = {
            singleton: self.views_dict[(singleton,)]
            for singleton in self.original_dataset.domain.attrs
        }

        synthesizer = UpdateConfig(
            self.attr_recode.dataset_recode.domain,
            self.args["num_synthesize_records"],
            update_config,
        )
        synthesizer.update.initialize_records(
            views_iterate_key,
            method=self.args["initialize_method"],
            singletons=singletons,
        )

        for update_iteration in range(self.args["update_iterations"]):
            self.logger.info("update round: %d" % (update_iteration,))

            synthesizer.update_alpha(update_iteration)
            # views_iterate_key = synthesizer.update_order(update_iteration, self.views_dict, views_iterate_key)

            for index, key in enumerate(views_iterate_key):
                self.logger.info(
                    "updating %s view: %s, num_key: %s"
                    % (index, key, self.views_dict[key].num_key)
                )

                synthesizer.update_records(self.views_dict[key], key, update_iteration)

        return synthesizer
