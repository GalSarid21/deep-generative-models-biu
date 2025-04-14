from common.configs.log_config import configure_log
from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType, PromptingMode

import experiments.runner as experiment_runner
import common.consts as consts

from argparse import Namespace

import logging
import pytest
import json


class ExperimentTest(AbstractExperiment):
    _TYPE = ExperimentType.TEST

    def __init__(self, args: Namespace) -> None:
        logging.info("Testing data downloading...")
        super().__init__(args)

    def run(self) -> None:
        logging.info("Running a poetry test experiment...")
        logging.info("Experiment is done!")


def get_test_cli_args() -> Namespace:
    return Namespace(
        experiment="test",
        hf_token=None,
        prompting_mode=PromptingMode.OPENBOOK,
        num_docs=consts.SUPPORTED_NUM_DOCS[0],
        model=consts.SUPPORTED_MODELS[0]
    )


def test_experiment_runner() -> None:
    configure_log()
    args = get_test_cli_args()
    logging.info(
        "Test environment Variables:\n" +
        json.dumps(vars(args), indent=4)
    )
    experiment_runner.run(args=args, running_cls=ExperimentTest)
