from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType

from argparse import Namespace

import logging


class TestExperiment(AbstractExperiment):
    _TYPE = ExperimentType.TEST

    def __init__(self, args: Namespace) -> None:
        logging.info("Testing data downloading...")
        super().__init__(args)

    def run(self) -> None:
        logging.info("Running a poetry test experiment...")
        logging.info("Experiment is done!")
