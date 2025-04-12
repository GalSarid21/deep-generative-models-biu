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

