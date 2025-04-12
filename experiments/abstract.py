from common.entities import ExperimentType
import common.nq_data as nq_data
import common.consts as consts

from argparse import Namespace
from typing import List
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

        nq_data.download_files(
            src_dir=consts.DATA_SRC_DIR,
            dst_dir=consts.DATA_DST_DIR,
            num_docs=args.num_docs
        )

    @classmethod
    def get_type(cls) -> ExperimentType:
        return cls._TYPE

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
