from common.entities import PromptingMode
from tests.conftest import download_nq_files_if_needed
import common.nq_data as nq_data
import common.consts as common_consts
import tests.consts as test_consts

from typing import Callable, Dict, Any
import logging
import shutil
import pytest
import json
import os


def test_data_downloading() -> None:

    if os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        shutil.rmtree(test_consts.DOCUMENTS_FOLDER_PATH)

    logging.info("Testing 'download_files' - downloading NQ data files")
    created_folder = nq_data.download_files(
        src_dir=common_consts.DATA_SRC_DIR,
        dst_dir=common_consts.DATA_DST_DIR,
        num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
    )
    assert created_folder == test_consts.DOCUMENTS_FOLDER_PATH

    created_files = [
        f for f in os.listdir(test_consts.DOCUMENTS_FOLDER_PATH)
            if os.path.isfile(
                os.path.join(test_consts.DOCUMENTS_FOLDER_PATH, f)
            )
    ]
    assert len(created_files) == len(test_consts.DOCUMETS_FOLDER_FILES)

    created_files.sort()
    for res_file, src_file in zip(
        created_files, test_consts.DOCUMETS_FOLDER_FILES
    ):
        assert res_file == src_file


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_skipping_files_downloading(
    test_results: Dict[str, Any],
    caplog: Callable
) -> None:

    download_nq_files_if_needed()
    with caplog.at_level(logging.INFO):
        nq_data.download_files(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )

        expected_log_msg_lines =\
            test_results["expected_log_msg"].strip().splitlines()
        caplog_lines = caplog.text.strip().splitlines()
        assert caplog_lines == expected_log_msg_lines


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_closedbook_documents_list_creation(
    test_results: Dict[str, Any]
) -> None:

    download_nq_files_if_needed()
    questions, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.CLOSEDBOOK
    )

    assert len(documents) == test_results["closedbook_num_docs"]
    assert len(questions) == test_results["closedbook_num_questions"]
    assert questions[0] == test_results["closedbook_question"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_openbook_documents_list_creation(
    test_results: Dict[str, Any]
) -> None:

    download_nq_files_if_needed()
    questions, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.OPENBOOK
    )

    test_idx = 1
    assert len(documents) == test_results["openbook_num_docs"]
    assert len(questions) == test_results["openbook_num_questions"]
    assert questions[test_idx] == test_results["openbook_question"]

    test_docs = documents[test_idx][:test_consts.NUM_DOCS_TO_TEST]
    for i, doc in enumerate(test_docs):
        assert doc.title == test_results["openbook_docs"][i]["title"]
        assert doc.text == test_results["openbook_docs"][i]["text"]
        assert doc.id == test_results["openbook_docs"][i]["id"]
        assert doc.score == test_results["openbook_docs"][i]["score"]
        assert doc.hasanswer == test_results["openbook_docs"][i]["hasanswer"]
        assert doc.isgold == test_results["openbook_docs"][i]["isgold"]
        assert doc.original_retrieval_index == \
            test_results["openbook_docs"][i]["original_retrieval_index"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_openbook_random_documents_list_creation(
    test_results: Dict[str, Any]
) -> None:
    
    download_nq_files_if_needed()
    questions, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.OPENBOOK_RANDOM
    )

    test_idx = 2
    assert len(documents) == test_results["openbook_random_num_docs"]
    assert len(questions) == test_results["openbook_random_num_questions"]
    assert questions[test_idx] == test_results["openbook_random_question"]

    test_doc = documents[test_idx][0]
    assert test_doc.title == test_results["openbook_random_docs"][0]["title"]
    assert test_doc.text == test_results["openbook_random_docs"][0]["text"]
    assert test_doc.id == test_results["openbook_random_docs"][0]["id"]
    assert test_doc.score == test_results["openbook_random_docs"][0]["score"]
    assert test_doc.hasanswer == test_results["openbook_random_docs"][0]["hasanswer"]
    assert test_doc.isgold == test_results["openbook_random_docs"][0]["isgold"]
    assert test_doc.original_retrieval_index == \
        test_results["openbook_random_docs"][0]["original_retrieval_index"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_nq_dict_creation(test_results: Dict[str, Any]) -> None:
    download_nq_files_if_needed()
    data = nq_data.read_files(
        folder_path=test_consts.DOCUMENTS_FOLDER_PATH,
        prompting_mode=PromptingMode.OPENBOOK
    )

    for ref_key, res_key in zip(
        sorted(data.keys()), test_results["data_dict_keys"]
    ):
        assert ref_key == res_key

    data_test_subset = {
        key: {
            "questions": [data[key]["questions"][0]],
            "documents": [data[key]["documents"][0][:2]]
        }
        for key in data.keys()
    }

    data_test_subset_json = json.dumps(
        data_test_subset, default=nq_data.serialize, indent=2
    )
    logging.info(f"Openbook Files Ditionary:\n{data_test_subset_json}")

    data_test_subset_json_dict = json.loads(data_test_subset_json)
    assert data_test_subset_json_dict == test_results["data_dict_subset"]


def download_nq_files_if_needed() -> None:
    if not os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        nq_data.download_files(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )
