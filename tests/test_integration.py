from common.configs.log_config import configure_log
from tests.mocks.vllm_wrapper import vLLMWrapperMock
from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType, PromptingMode
from src.metrics import best_subspan_em

import experiments.runner as experiment_runner
import common.consts as consts

from argparse import Namespace

import logging
import json


class ExperimentTest(AbstractExperiment):
    _TYPE = ExperimentType.TEST

    def __init__(self, args: Namespace) -> None:
        logging.info("Testing data downloading...")

        self._llm = vLLMWrapperMock(
            model=args.model,
            dtype=args.dtype,
            num_gpus=args.num_gpus
        )

        super().__init__(args)

    def run(self) -> None:
        logging.info("Running a poetry test experiment...")

        data_key = list(self._data.keys())[0]
        logging.info(f"Using data key: {data_key}")

        question = self._data[data_key]["questions"][0]
        documents = self._data[data_key]["documents"][0]
        prompt = self._prompt_builder.build(question, documents)
        logging.info(f"Prompt:\n{prompt}")
        
        model_answer = self._llm.generate(prompt, **self._sampling_params)
        logging.info(f"Model answer:\n{model_answer}")

        ground_truths = self._data[data_key]["answers"][0]
        score = best_subspan_em(
            prediction=model_answer, ground_truths=ground_truths
        )
        logging.info(f"Model answer best_subspan_em score: {score}")

        self._add_new_result_entry(prompt, data_key, model_answer, score)
        logging.info(
            f"Results object with new entry:\n" +
            json.dumps(self._results, indent=2)
        )

        logging.info("Experiment is done!")


def get_test_cli_args() -> Namespace:
    return Namespace(
        experiment="test",
        hf_token=None,
        prompting_mode=PromptingMode.OPENBOOK,
        num_docs=consts.SUPPORTED_NUM_DOCS[0],
        model=consts.SUPPORTED_MODELS[0],
        dtype=consts.SUPPORTED_DTYPES[0],
        num_gpus=consts.DEFAULT_NUM_GPUS,
        temperature=consts.DEFAULT_TEMPERATURE,
        top_p=consts.DEFAULT_TOP_P,
        max_tokens=consts.DEFAULT_MAX_TOKENS
    )


def test_experiment_runner() -> None:
    configure_log()
    args = get_test_cli_args()
    logging.info(
        "Test environment Variables:\n" +
        json.dumps(vars(args), indent=4)
    )
    experiment_runner.run(args=args, running_cls=ExperimentTest)
