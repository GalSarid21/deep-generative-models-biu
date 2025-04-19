from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType, PromptingMode
from argparse import Namespace
import common.nq_data as nq_data
import common.consts as consts

from datetime import datetime, UTC
import logging
import json
import os


class GoldIdxChange(AbstractExperiment):
    _TYPE = ExperimentType.GOLD_IDX_CHANGE

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        
        created_folder = nq_data.download_files_by_num_docs(
            src_dir=consts.DATA_SRC_DIR,
            dst_dir=consts.DATA_DST_DIR,
            num_docs=args.num_docs
        )

        self._data = nq_data.read_files_by_num_docs(
            folder_path=created_folder,
            prompting_mode=self._prompting_mode
        )

        if args.test_mode is True:
            n = consts.TEST_NUM_EXAMPLES
            logging.info(f"Argument test_mode=True. Truncating data to {n}.")
            self._truncate_data(n)

        self._results = self._get_empty_results_dict(args)

    def _log_experiment_results(self) -> None:
        logging.info(f"Logging test results.")

        # taking only the model name without HF repo name
        # for example: tiiuae/Falcon3-Mamba-7B-Instruct --> 
        # tiiuae/Falcon3-Mamba-7B-Instruct
        model_short = self._results["model"].split("/")[-1]
        experiment_type = self._results["experiment_type"].replace("-", "_")
        num_docs = self._results["num_documents"]
        prompting_mode = self._results["prompting_mode"].replace("-", "_")
        timestamp = int(datetime.now(UTC).timestamp())

        result_file_dir = f"{consts.RESULTS_DIR}/{model_short}/" \
            + f"{experiment_type}_experiment/" \
            + f"{prompting_mode}_prompting_mode/"

        if prompting_mode is not PromptingMode.CLOSEDBOOK:
            result_file_dir += f"/{num_docs}_docs"

        os.makedirs(result_file_dir, exist_ok=True)
        result_file_path = f"{result_file_dir}/{timestamp}.json"
        with open(result_file_path, "w") as f:
            f.write(json.dumps(self._results, indent=2) + "\n")

        logging.info(f"Results saved to {result_file_path}")
