from common.entities import Document, PromptingMode

from pathlib import Path
from typing import List, Tuple, Dict, Any
from random import shuffle
from xopen import xopen
from copy import deepcopy
from tqdm import tqdm

import logging
import shutil
import gzip
import json
import os


def download_files(
    num_docs: int,
    dst_dir: str,
    src_dir: str
) -> None:
    """
    Downloads the NQ dataset files from the lost-in-the-middle local cloned
    repo at `src_dir` and saves it under `dst_dir`.
    """
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


def read_files(
    folder_path: str,
    prompting_mode: PromptingMode
) -> Dict[str, List[List[Document]]]:

    folder_path = Path(folder_path)
    jsonl_files = list(folder_path.glob("*.jsonl"))
    files_data = {}
    for jsonl in jsonl_files:
        questions, documents = read_file(
            file_path=jsonl, prompting_mode=prompting_mode
        )
        # stem should look like the following:
        # nq-open-10_total_documents_gold_at_0
        # hence - split("_documents_")[-1] - will
        # create a short name such as "gold_at_0"
        file_short_name = jsonl.stem.split("_documents_")[-1]
        files_data.update({
            file_short_name: {"questions": questions, "documents": documents}
        })

    return files_data


def read_file(
    file_path: str,
    prompting_mode: PromptingMode
) -> Tuple[List[str], List[List[Document]]]:
    """
    Reads NQ dataset file using `file_path` and creates a list of lists
    of documents with a matching list of questions.
    """
    all_questions = []
    all_documents = []

    with xopen(file_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            question = input_example["question"]
            all_questions.append(question)
            if prompting_mode is PromptingMode.CLOSEDBOOK:
                # closedbook doesn not need context document - 
                # we're returning and empty list
                continue
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if prompting_mode is PromptingMode.OPENBOOK_RANDOM:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            all_documents.append(documents)

    return all_questions, all_documents


def data_serializer(obj: Any) -> Dict[str, str]:
    if isinstance(obj, Document):
        return obj.to_dict()
    elif (
        isinstance(obj, list) and
        isinstance(obj[0], list) and
        isinstance(obj[0][0], Document)
    ):
        return [
            [doc.to_dict() for doc in sublist]
            for sublist in obj
        ]
    else:
        return obj
