from src.prompt_builder import PromptBuilder
from common.entities import PromptingMode, Document
from tests.conftest import download_nq_files_if_needed
import common.nq_data as nq_data
import tests.consts as test_consts

from typing import List, Tuple, Dict, Any
import logging
import pytest


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/prompt_building.yaml"],
    indirect=True
)
def test_openbook_prompt_builder(test_results: Dict[str, Any]) -> None:
    prompt = _get_test_prompt_with_documents(
        prompting_mode=PromptingMode.OPENBOOK,
        test_idx=0
    )
    assert prompt == test_results["openbook"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/prompt_building.yaml"],
    indirect=True
)
def test_openboom_random_prompt_builder(test_results: Dict[str, Any]) -> None:
    prompt = _get_test_prompt_with_documents(
        prompting_mode=PromptingMode.OPENBOOK_RANDOM,
        test_idx=1
    )
    prompt_lines = prompt.strip().splitlines()
    res_lines = test_results["openbook_random"].strip().splitlines()
    
    assert len(res_lines) == len(prompt_lines)
    # opening line
    assert res_lines[0] == prompt_lines[0]
    # question line
    assert res_lines[-2] == prompt_lines[-2]
    # "Answer:" line
    assert res_lines[-1] == prompt_lines[-1]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/prompt_building.yaml"],
    indirect=True
)
def test_closedbook_prompt_builder(test_results: Dict[str, Any]) -> None:
    download_nq_files_if_needed()
    prompting_mode = PromptingMode.CLOSEDBOOK
    questions, documents_lists = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=prompting_mode
    )

    # on closedbook we aren't suppose to get documents_lists.
    assert len(documents_lists) == 0

    builder = PromptBuilder(prompting_mode)
    prompt = builder.build(
        question=questions[0],
        documents=documents_lists
    )

    logging.info(f"Test Prompt [{prompting_mode.value}]:\n{prompt}")
    assert prompt == test_results["closedbook"]


def _get_test_prompt_with_documents(
    prompting_mode: PromptingMode,
    test_idx: int
) -> Tuple[List[str], List[List[Document]]]:

    download_nq_files_if_needed()
    questions, documents_lists = nq_data.read_file(
        file_path=test_consts.TEST_DOCUMENT_PATH,
        prompting_mode=prompting_mode
    )

    builder = PromptBuilder(prompting_mode)
    prompt = builder.build(
        question=questions[test_idx],
        documents=documents_lists[test_idx]
    )

    logging.info(f"Test Prompt [{prompting_mode.value}]:\n{prompt}")
    return prompt
