from common.env.arg_setting import set_hf_token
from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType
from experiments import ALL_EXPERIMENTS

import common.consts as consts

from argparse import Namespace
from typing import Type


class ExperimentRunner:

    def __init__(self, args: Namespace) -> None:
        self._args = args
        set_hf_token(self._args.hf_token)

    def run(self) -> None:
        experiment_cls = self._get_experiment_class()
        experiment = experiment_cls(args=self._args)
        experiment.run()

    def _get_experiment_class(self) -> Type[AbstractExperiment]:
        try:
            experiment_type = ExperimentType(self._args.experiment)
        except Exception:
            # raise readable custom error
            raise ValueError(
                consts.INVALID_ENUM_CREATION_MSG.format(
                    obj=ExperimentType, arg=self._args.experiment
                )
            )

        for experiment_cls in ALL_EXPERIMENTS:
            if experiment_cls.get_type() == experiment_type:
                return experiment_cls

        raise ValueError(f"Unknown experiment type: {experiment_type}")
