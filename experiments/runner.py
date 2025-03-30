from common.env_utils.arg_setting import set_hf_token
from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType
from experiments import ALL_EXPERIMENTS

import common.consts as consts

from argparse import Namespace
from typing import Type


def run(args: Namespace) -> None:
    experiment_cls = _get_experiment_class(args)
    experiment = experiment_cls(args)
    experiment.run()


def _get_experiment_class(args: Namespace) -> Type[AbstractExperiment]:
    try:
        experiment_type = ExperimentType(args.experiment)
    except Exception:
        # raise readable custom error
        raise ValueError(
            consts.INVALID_ENUM_CREATION_MSG.format(
                obj=ExperimentType, arg=args.experiment
            )
        )

    for experiment_cls in ALL_EXPERIMENTS:
        if experiment_cls.get_type() == experiment_type:
            return experiment_cls

    raise ValueError(f"Unknown experiment type: {experiment_type}")
