from common.configs.log_config import configure_log
from common.entities import ExperimentType, PromptingMode
from experiments import GoldIdxChange, NumDocsChange, AbstractExperiment
import experiments.runner as experiment_runner
import common.consts as common_consts
import tests.consts as test_consts

from argparse import Namespace
from typing import Optional
import logging
import json


def test_gold_idx_experiment_with_instruct_model() -> None:
    args = _get_test_cli_args(
        experiment=ExperimentType.GOLD_IDX_CHANGE.value,
        num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
    )
    _run_e2e_test(args=args, running_cls=GoldIdxChange)


def test_gold_idx_experiment_with_base_model() -> None:
    args = _get_test_cli_args(
        model=test_consts.NON_CHAT_TEST_MODEL,
        experiment=ExperimentType.GOLD_IDX_CHANGE.value,
        num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
    )
    _run_e2e_test(args=args, running_cls=GoldIdxChange)


def test_num_docs_experiment() -> None:
    args = _get_test_cli_args(
        experiment=ExperimentType.NUM_DOCS_CHANGE.value,
        gold_idx=common_consts.SUPPORTED_GOLD_IDXS[0]
    )
    _run_e2e_test(args=args, running_cls=NumDocsChange)


def _run_e2e_test(args: Namespace, running_cls: AbstractExperiment) -> None:
    configure_log()
    logging.info(
        "Test environment Variables:\n" +
        json.dumps(vars(args), indent=2)
    )
    experiment_runner.run(args, running_cls)


def _get_test_cli_args(
    experiment: str,
    model: Optional[str] = None,
    num_docs: Optional[int] = None,
    gold_idx: Optional[int] = None
) -> Namespace:

    return Namespace(
        experiment=experiment,
        hf_token=None,
        prompting_mode=PromptingMode.OPENBOOK,
        num_docs=num_docs,
        gold_idx=gold_idx,
        model=model or common_consts.DEFAULT_MODEL,
        dtype=common_consts.SUPPORTED_DTYPES[0],
        num_gpus=common_consts.DEFAULT_NUM_GPUS,
        temperature=common_consts.DEFAULT_TEMPERATURE,
        top_p=common_consts.DEFAULT_TOP_P,
        max_tokens=common_consts.DEFAULT_MAX_TOKENS,
        max_model_len=common_consts.DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization=common_consts.DEFAULT_MAX_GPU_UTIL,
        results_dir=None,
        test_mode=True
    )
