from common.configs.log_config import configure_log
from common.entities import ExperimentType, PromptingMode
from experiments import GoldIdxChange
import experiments.runner as experiment_runner
import common.consts as consts

from argparse import Namespace
from typing import Optional
import logging
import json


def get_test_cli_args(
    experiment: str,
    num_docs: Optional[int] = None,
    gold_idx: Optional[int] = None
) -> Namespace:

    return Namespace(
        experiment=experiment,
        hf_token=None,
        prompting_mode=PromptingMode.OPENBOOK,
        num_docs=num_docs,
        gold_idx=gold_idx,
        model=consts.SUPPORTED_MODELS[0],
        dtype=consts.SUPPORTED_DTYPES[0],
        num_gpus=consts.DEFAULT_NUM_GPUS,
        temperature=consts.DEFAULT_TEMPERATURE,
        top_p=consts.DEFAULT_TOP_P,
        max_tokens=consts.DEFAULT_MAX_TOKENS,
        test_mode=True
    )


def test_gold_idx_experiment() -> None:
    configure_log()
    args = get_test_cli_args(
        experiment=ExperimentType.GOLD_IDX_CHANGE.value,
        num_docs=consts.SUPPORTED_NUM_DOCS[0]
    )

    logging.info(
        "Test environment Variables:\n" +
        json.dumps(vars(args), indent=2)
    )

    experiment_runner.run(args=args, running_cls=GoldIdxChange)
