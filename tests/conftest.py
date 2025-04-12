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


def download_nq_files_if_needed() -> None:
    if not os.path.exists(test_consts.DOCUMENTS_FOLDER_PATH):
        nq_data.download_files(
            src_dir=common_consts.DATA_SRC_DIR,
            dst_dir=common_consts.DATA_DST_DIR,
            num_docs=common_consts.SUPPORTED_NUM_DOCS[0]
        )
