from tests.mocks.vllm_wrapper import vLLMWrapperMock
from src.prompt_builder import PromptBuilder
from common.entities import ExperimentType, PromptingMode
from src.wrappers import HfTokenizer, vLLMWrapper
from src.metrics import best_subspan_em
import common.consts as consts

from argparse import Namespace
from typing import List, Dict, Union, Optional, Tuple, Any
from xopen import xopen
from abc import ABC, abstractmethod
import datetime
import logging
import torch
import json
import os


class AbstractExperiment(ABC):
    _TYPE = None

    @abstractmethod
    def run(self) -> None:
        pass

    def __init__(self, args: Namespace) -> None:
        if torch.cuda.is_available():
            msg = f"HW type: GPU | HW name: {torch.cuda.get_device_name(0)}"
        else:
            import platform
            msg = f"HW type: CPU | HW name: {platform.processor()}"
        logging.info(msg)

        self._create_process_dirs(
            dirs=[consts.RESULTS_DIR, consts.DATA_DST_DIR]
        )

        self._prompting_mode = PromptingMode(args.prompting_mode)
        self._tokenizer = HfTokenizer(args.model)
        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer
        )

        self._llm = self._load_llm(args)
        self._sampling_params = self._get_llm_sampling_params(args)

        self._results_dir = args.results_dir or consts.RESULTS_DIR
        self._data = None
        self._results = None

    @property
    def results(self) -> Dict[str, Any]:
        return self._results

    @classmethod
    def get_type(cls) -> ExperimentType:
        return cls._TYPE

    def _load_llm(
        self,
        args: Namespace
    ) -> Union[vLLMWrapper, vLLMWrapperMock]:

        vllm_payload = {
            "model": args.model,
            "dtype": args.dtype,
            "num_gpus": args.num_gpus,
            "max_model_len": args.max_model_len
        }

        if args.test_mode is True:
            logging.info(f"Argument test_mode=True. Loading mock llm.")
            return vLLMWrapperMock(**vllm_payload)
        return vLLMWrapper(**vllm_payload)

    def _add_new_result_entries(
        self,
        prompts: List[str],
        model_answers: List[str],
        scores: List[float],
        metric: str,
        key: str
    ) -> None:

        num_prompt_tokens_list = [
            self._tokenizer.count_tokens(
                prompt=prompt, prompt_with_inst_tokens=True
            ) for prompt in prompts
        ]

        experiment = self._results["experiments"][key]
        experiment["model_answers"].extend(model_answers)
        experiment["scores"].extend(scores)
        experiment["metric"] = metric
        experiment["num_prompt_tokens"].extend(num_prompt_tokens_list)

    def _create_process_dirs(self, dirs: List[str]) -> None:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def _get_llm_sampling_params(self, args: Namespace) -> Dict[str, Any]:
        return {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p
        }

    def _get_prompts_by_data_key(self, key: str) -> List[str]:
        questions = self._data[key]["questions"]

        if self._prompting_mode is not PromptingMode.CLOSEDBOOK:
            documents_list = self._data[key]["documents"]
            return [
                self._prompt_builder.build(question, documents)
                for question, documents in zip(questions, documents_list)
            ]
        return [
            self._prompt_builder.build(question)
            for question in questions
        ]

    def _calc_predictions_scores(
        self,
        predictions: List[str],
        key: str
    ) -> Tuple[str, List[float]]:

        answers_list = self._data[key]["answers"]
        scores = [
            best_subspan_em(prediction=prediction, ground_truths=answers)
            for prediction, answers in zip(predictions, answers_list)
        ]
        return "best_subspan_em", scores

    def _truncate_data(
        self,
        n: Optional[int] = 1,
        in_place: Optional[bool] = True
    ) -> Optional[Dict[str, Any]]:
        """
        Truncates each of the 'questions', 'answers', and 'documents' lists in the dataset
        to the first `n` elements.

        Args:
            data_dict (dict): The original dataset.
            n (int): The number of examples to keep in each list.
            in_place (bool): If True, modifies the original dict. If False, returns a new truncated dict.

        Returns:
            dict: The truncated dataset (only if in_place=False).
        """
        target = self._data if in_place is True else {}

        for key, entry in self._data.items():
            if in_place is True:
                entry["questions"] = entry["questions"][:n]
                entry["answers"] = entry["answers"][:n]
                entry["documents"] = entry["documents"][:n]
            else:
                target[key] = {
                    "questions": entry["questions"][:n],
                    "answers": entry["answers"][:n],
                    "documents": entry["documents"][:n]
                }

        if in_place is False:
            return target

    def _log_experiment_results(self) -> None:
        logging.info(f"Logging test results.")

        # taking only the model name without HF repo name
        # for example: tiiuae/Falcon3-Mamba-7B-Instruct --> 
        # tiiuae/Falcon3-Mamba-7B-Instruct
        model_short = self._results["model"].split("/")[-1]
        experiment_type = self._results["experiment_type"]
        num_docs = self._results["num_documents"]
        prompting_mode = self._results["prompting_mode"]
        timestamp = int(datetime.datetime.now(datetime.UTC).timestamp())

        result_file_dir = f"{consts.RESULTS_DIR}/{model_short}/" \
            + f"{experiment_type}_experiment/{num_docs}_docs/" \
            + f"{prompting_mode}_prompting_mode"

        os.makedirs(result_file_dir, exist_ok=True)
        result_file_path = f"{result_file_dir}/{timestamp}.json"
        with xopen(result_file_path, "w") as f:
            f.write(json.dumps(self._results, indent=2) + "\n")

        logging.info(f"Results saved to {result_file_path}")
