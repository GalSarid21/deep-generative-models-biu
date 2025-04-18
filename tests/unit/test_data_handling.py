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


def test_download_files_by_gold_idx() -> None:

    for num_docs in common_consts.SUPPORTED_NUM_DOCS:
        file_path =\
            test_consts.DOCUMENTS_FOLDER_TEMPLATE.format(num_docs=num_docs)
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    logging.info("Testing 'download_files_by_gold_idx' - downloading NQ data files")
    gold_idx = common_consts.SUPPORTED_GOLD_IDXS[0]
    created_folders = nq_data.download_files_by_gold_idx(
        src_dir=common_consts.DATA_SRC_DIR,
        dst_dir=common_consts.DATA_DST_DIR,
        gold_idx=gold_idx
    )

    created_folders.sort()
    for created_folder, num_docs in zip(
        created_folders, common_consts.SUPPORTED_NUM_DOCS
    ):
        assert created_folder == test_consts.DOCUMENTS_FOLDER_TEMPLATE.format(
            num_docs=num_docs
        )

        created_files = [
            f for f in os.listdir(created_folder)
                if os.path.isfile(os.path.join(created_folder, f))
        ]
        # we expect only the relevant golden index file to be created
        assert len(created_files) == 1
        assert created_files[0] == test_consts.DOCUMENT_NAME_TEMPLATE.format(
            num_docs=num_docs, gold_idx=gold_idx
        )


def test_download_files_by_num_docs() -> None:

    if os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        shutil.rmtree(test_consts.DOCUMENTS_FOLDER_PATH)

    logging.info("Testing 'download_files_by_num_docs' - downloading NQ data files")
    created_folder = nq_data.download_files_by_num_docs(
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
        # skipping on existing file
        nq_data.download_files_by_num_docs(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )

        # skipping on incorrect gold_idx
        nq_data.download_files_by_gold_idx(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            gold_idx=common_consts.SUPPORTED_GOLD_IDXS[-1] #29
        )

        assert caplog.text.strip() == test_results["expected_log_msg"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_closedbook_documents_list_creation(
    test_results: Dict[str, Any]
) -> None:

    download_nq_files_if_needed()
    questions, answers, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.CLOSEDBOOK
    )

    assert len(documents) == test_results["closedbook_num_docs"]
    assert len(answers) == test_results["closedbook_num_answers"]
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
    questions, answers, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.OPENBOOK
    )

    test_idx = 1
    assert len(documents) == test_results["openbook_num_docs"]
    assert len(answers) == test_results["openbook_num_answers"]
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
    questions, answers, documents = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=PromptingMode.OPENBOOK_RANDOM
    )

    test_idx = 2
    assert len(documents) == test_results["openbook_random_num_docs"]
    assert len(answers) == test_results["openbook_random_num_answers"]
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
def test_gold_idx_change_data_dict_creation(
    test_results: Dict[str, Any]
) -> None:

    download_nq_files_if_needed()
    # gold index experiment needs all of the json files from
    # a specific folder, by the number of documents
    data = nq_data.read_files_by_num_docs(
        folder_path=test_consts.DOCUMENTS_FOLDER_PATH,
        prompting_mode=PromptingMode.OPENBOOK
    )

    for ref_key, res_key in zip(
        sorted(data.keys()), test_results["gold_idx_data_dict_keys"]
    ):
        assert ref_key == res_key

    data_test_subset = {
        key: {
            "questions": [data[key]["questions"][0]],
            "answers": [data[key]["answers"][0]],
            "documents": [data[key]["documents"][0][:2]]
        }
        for key in data.keys()
    }

    data_test_subset_json = json.dumps(
        data_test_subset, default=nq_data.serialize, indent=2
    )
    logging.info(f"Openbook Files Dictionary [gold_idx_change]:\n{data_test_subset_json}")

    data_test_subset_json_dict = json.loads(data_test_subset_json)
    assert data_test_subset_json_dict == test_results["gold_idx_data_dict_subset"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/data_handling.yaml"],
    indirect=True
)
def test_num_docs_change_data_dict_creation(
    test_results: Dict[str, Any]
) -> None:
    # num docs experiment needs the relevant json file from
    # each of the folders, by the gold index
    data = nq_data.read_files_by_gold_idx(
        folder_paths=[
            test_consts.DOCUMENTS_FOLDER_TEMPLATE.format(
                num_docs=num_docs
            ) for num_docs in common_consts.SUPPORTED_NUM_DOCS
        ],
        prompting_mode=PromptingMode.OPENBOOK,
        gold_idx=common_consts.SUPPORTED_GOLD_IDXS[0]
    )

    for ref_key, res_key in zip(
        sorted(data.keys()), test_results["num_docs_data_dict_keys"]
    ):
        assert ref_key == res_key

    data_test_subset = {
        key: {
            "questions": [data[key]["questions"][0]],
            "answers": [data[key]["answers"][0]],
            "documents": [data[key]["documents"][0][:2]]
        }
        for key in data.keys()
    }

    data_test_subset_json = json.dumps(
        data_test_subset, default=nq_data.serialize, indent=2
    )
    logging.info(
        f"Openbook Files Dictionary [num_docs_change]:\n{data_test_subset_json}"
    )

    data_test_subset_json_dict = json.loads(data_test_subset_json)
    assert data_test_subset_json_dict == test_results["num_docs_data_dict_subset"]


def download_nq_files_if_needed() -> None:
    if not os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        nq_data.download_files_by_num_docs(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )
