from experiments.abstract import AbstractExperiment
from experiments.test import TestExperiment
from common.entities import ExperimentType

from argparse import Namespace


class ExperimentRunner:

    def __init__(self, args: Namespace) -> None:
        self._args = args
        # mapping dict to choose experiment dynamically at run time
        self._experiment_mapping = {
            ExperimentType.TEST: TestExperiment
        }

    def run(self) -> None:
        experiment = self._get_running_experiment()
        experiment.run()

    def _get_running_experiment(self) -> AbstractExperiment:
        # invalid experiment would raise an error
        experiment_type = ExperimentType(self._args.experiment)
        experiment_class = self._experiment_mapping[experiment_type]
        return experiment_class()
