from common.entities import ExperimentType
import common.consts as consts

from argparse import ArgumentParser, Namespace
import logging
import json


def read_cli_env_args() -> Namespace:
    parser = ArgumentParser("")

    parser.add_argument(
        "--experiment",
        help="experiment type to run.",
        type=str,
        choices=[et.value for et in ExperimentType]
    )

    parser.add_argument(
        "--hf_token",
        help="HF Token for model downloading.",
        type=str
    )

    parser.add_argument(
        "--num_docs",
        help="number of documents to use in the experiment.",
        type=int,
        choices=consts.SUPPORTED_NUM_DOCS,
        default=consts.SUPPORTED_NUM_DOCS[0]
    )

    parser.add_argument(
        "--model",
        help="HF model repo to use in experiment.",
        type=str,
        choices=consts.SUPPORTED_MODELS,
        default=consts.SUPPORTED_MODELS[0]
    )

    parser.add_argument(
        "--prompting_mode",
        help="experiment type to run.",
        type=str,
        choices=[et.value for et in ExperimentType]
    )

    args = parser.parse_args()
    logging.info(
        "Environment Variables:\n" +
        json.dumps(vars(args), indent=4)
    )
    return args
