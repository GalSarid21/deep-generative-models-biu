from common.entities import ExperimentType
from common.consts import Consts

from argparse import ArgumentParser, Namespace

import logging
import json


class CliEnvArgs:
    
    @staticmethod
    def get_args() -> Namespace:
        parser = ArgumentParser("")

        parser.add_argument(
            "--experiment",
            help="experiment type to run.",
            type=str,
            choices=[et.value for et in ExperimentType]
        )

        parser.add_argument(
            "--num_docs",
            help="number of documents to use in the experiment.",
            type=int,
            choices=Consts.SUPPORTED_NUM_DOCS,
            default=Consts.SUPPORTED_NUM_DOCS[0]
        )

        parser.add_argument(
            "--model",
            help="HF model repo to use in experiment.",
            type=str,
            choices=Consts.SUPPORTED_MODELS,
            default=Consts.SUPPORTED_MODELS[0]
        )

        args = parser.parse_args()
        logging.info(
            "Environment Variables:\n" +
            json.dumps(vars(args), indent=4)
        )
        return args
