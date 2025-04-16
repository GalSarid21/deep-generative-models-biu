from experiments.abstract import AbstractExperiment
from common.entities import ExperimentType
from argparse import Namespace
import common.nq_data as nq_data
import common.consts as consts

from datetime import datetime
import logging


class GoldIdxChange(AbstractExperiment):
    _TYPE = ExperimentType.GOLD_IDX_CHANGE

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        
        created_folder = nq_data.download_files_by_num_docs(
            src_dir=consts.DATA_SRC_DIR,
            dst_dir=consts.DATA_DST_DIR,
            num_docs=args.num_docs
        )

        self._data = nq_data.read_folder_files(
            folder_path=created_folder,
            prompting_mode=self._prompting_mode
        )

        if args.test_mode is True:
            n = consts.TEST_NUM_EXAMPLES
            logging.info(f"Argument test_mode=True. Truncating data to {n}.")
            self._truncate_data(n)

        self._results = self._get_empty_results_dict(args)

    def run(self) -> None:
        logging.info("Running a gold index change experiment...")
        for key in self._data.keys():
            logging.info(f"Starting process '{key}'...")

            prompts = self._get_prompts_by_data_key(key)
            predictions = self._llm.generate_batch(
                prompts, **self._sampling_params
            )
            scores = self._calc_predictions_scores(predictions, key)

            self._add_new_result_entries(
                prompts=prompts,
                model_answers=predictions,
                scores=scores,
                key=key
            )

        self._log_experiment_results()

    def _get_empty_results_dict(
        self,
        args: Namespace
    ) -> None:

        results = {
            "model": args.model,
            "experiment_type": self._TYPE.value,
            "num_documents": args.num_docs,
            "prompting_mode": self._prompting_mode.value,
            "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiments": {}
        }

        experiments = results["experiments"]
        for key in self._data.keys():
            experiments.update({
                key: {
                    "model_answers": [],
                    # [{"value": 1.0, "metric": "best_subset_em"}]
                    "scores": [],
                    "num_prompt_tokens": []
                }
            })

        return results
