from src.wrappers.hf_tokenizer import HfTokenizer
from src.prompt_builder import PromptBuilder
from common.entities import ExperimentType, PromptingMode
import common.nq_data as nq_data
import common.consts as consts

from argparse import Namespace
from datetime import datetime
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import logging
import shutil
import gzip
import os


class AbstractExperiment(ABC):
    _TYPE = None

    @abstractmethod
    def run(self) -> None:
        pass

    def __init__(self, args: Namespace) -> None:
        self._create_process_dirs(
            dirs=[consts.RESULTS_DIR, consts.DATA_DST_DIR]
        )

        created_folder = nq_data.download_files(
            src_dir=consts.DATA_SRC_DIR,
            dst_dir=consts.DATA_DST_DIR,
            num_docs=args.num_docs
        )

        self._prompting_mode = PromptingMode(args.prompting_mode)
        self._data = nq_data.read_files(
            folder_path=created_folder,
            prompting_mode=self._prompting_mode
        )

        self._tokenizer = HfTokenizer(args.model)
        self._prompt_builder = PromptBuilder(self._prompting_mode)

        # vLLM sampling params
        self._sampling_params = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p
        }

        # results object
        results = {
            "experiment_type": self._TYPE.value,
            "num_documents": args.num_docs,
            "prompting_mode": self._prompting_mode.value,
            "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for key in self._data.keys():
            results.update({
                key: {
                    "model_answers": [],
                    # [{"value": 1.0, "metric": "best_subset_em"}]
                    "scores": [],
                    "num_prompt_tokens": []
                }
            })
        self._results = results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results

    @classmethod
    def get_type(cls) -> ExperimentType:
        return cls._TYPE

    def _add_new_result_entry(
        self,
        prompt: str,
        key: str,
        model_answer: str,
        score: float
    ) -> None:

        num_prompt_tokens = self._tokenizer.count_tokens(prompt=prompt)
        self._results[key]["model_answers"].append(model_answer)
        self._results[key]["scores"].append(score)
        self._results[key]["num_prompt_tokens"].append(num_prompt_tokens)

    def _create_process_dirs(self, dirs: List[str]) -> None:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def _download_experiment_data_files(
        self,
        num_docs: int,
        dst_dir: str,
        src_dir: str
    ) -> None:

        logging.info(f"Downloading NQ Data...")
        data_folders = [
            f for f in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, f))
                and f.startswith(str(num_docs))
        ]

        for folder in data_folders:
            src_folder = f"{src_dir}/{folder}"
            dst_folder = f"{dst_dir}/{folder}"
            os.makedirs(dst_folder, exist_ok=True)

            for file_name in os.listdir(src_folder):
                if file_name.endswith(".gz"):
                    src_file = f"{src_folder}/{file_name}"
                    dst_file = f"{dst_folder}/{file_name.replace('.gz', '')}"

                    if not os.path.exists(dst_file):
                        with(
                            gzip.open(src_file, "rb") as f_in,
                            open(dst_file, "wb") as f_out
                        ):
                            logging.info(f"Downloading file: {src_file} to: {dst_file}")
                            shutil.copyfileobj(f_in, f_out)

                    else:
                        logging.info(f"Skipping existing file: {src_file}")
