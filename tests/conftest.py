from tests.mocks.vllm_wrapper import vLLMWrapperMock
from src.wrappers import HfTokenizer
import common.nq_data as nq_data
import common.consts as common_consts
import tests.consts as test_consts

from typing import Dict, Any
import pytest
import yaml
import os


@pytest.fixture
def test_results(request: pytest.FixtureRequest) -> Dict[str, Any]:

    res_file = request.param
    if not res_file:
        pytest.fail("Missing 'res_file_name'", pytrace=False)

    if (
        not res_file.endswith(".yaml") and
        not res_file.endswith(".yml")
    ):
        res_file = f"{res_file}.yaml"

    if not os.path.exists(res_file):
        pytest.fail(
            f"Test results file '{res_file}' not found.",
            pytrace=False
        )

    with open(res_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


@pytest.fixture
def vllm_wrapper_mock() -> vLLMWrapperMock:
    return vLLMWrapperMock(
        model=common_consts.SUPPORTED_MODELS[0],
        dtype=common_consts.SUPPORTED_DTYPES[0],
        num_gpus=common_consts.DEFAULT_NUM_GPUS
    )


@pytest.fixture
def hf_tokenizer(request) -> Dict[str, Any]:

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        pytest.fail("Missing 'hf_token'", pytrace=False)

    model = request.param
    if model is None:
        pytest.fail("Missing 'model'", pytrace=False)
    return HfTokenizer(model)


def download_nq_files_if_needed() -> None:
    if not os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        nq_data.download_files_by_num_docs(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )
