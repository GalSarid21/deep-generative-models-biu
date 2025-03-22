from experiments.abstract import AbstractExperiment

import logging


class TestExperiment(AbstractExperiment):

    def __init__(self) -> None:
        pass

    def run(self) -> None:
        logging.info("Running a poetry test experiment...")
        logging.info("Experiment is done!")
