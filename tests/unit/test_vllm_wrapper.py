from tests.mocks.vllm_wrapper import vLLMWrapperMock
import common.consts as common_consts

from typing import Callable, Dict, Any
import logging
import pytest


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/vllm_wrapper.yaml"],
    indirect=True
)
def test_obj_initiation(
    test_results: Dict[str, Any],
    caplog: Callable
) -> None:

    with caplog.at_level(logging.INFO):
        _ = vLLMWrapperMock(
            model=common_consts.SUPPORTED_MODELS[0],
            dtype=common_consts.SUPPORTED_DTYPES[0],
            num_gpus=common_consts.DEFAULT_NUM_GPUS,
            max_model_len=common_consts.DEFAULT_MAX_MODEL_LEN
        )
    assert caplog.text.strip() == test_results["obj_initiation"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/vllm_wrapper.yaml"],
    indirect=True
)
def test_generate(
    test_results: Dict[str, Any],
    vllm_wrapper_mock: vLLMWrapperMock
) -> None:

    prompt = test_results["prompt"]
    model_answer = vllm_wrapper_mock.generate(
        prompt=prompt,
        top_p=common_consts.DEFAULT_TOP_P,
        max_tokens=common_consts.DEFAULT_MAX_TOKENS,
        temperature=common_consts.DEFAULT_TEMPERATURE
    )
    assert isinstance(model_answer, str)

    logging.info(
        f"{vllm_wrapper_mock.__class__.__name__} model answer:\n{model_answer}"
    )

    resp_prefix = test_results["mock_resp_prefix"]
    resp = f"{resp_prefix}{prompt[:common_consts.DEFAULT_MAX_TOKENS]}..."
    assert resp == model_answer


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/vllm_wrapper.yaml"],
    indirect=True
)
def test_generate_batch(
    test_results: Dict[str, Any],
    vllm_wrapper_mock: vLLMWrapperMock
) -> None:

    prompts = test_results["batch_prompts"]
    model_answers = vllm_wrapper_mock.generate_batch(
        prompts=prompts,
        top_p=common_consts.DEFAULT_TOP_P,
        max_tokens=common_consts.DEFAULT_MAX_TOKENS,
        temperature=common_consts.DEFAULT_TEMPERATURE
    )
    assert isinstance(model_answers, list)

    logging.info(
        f"{vllm_wrapper_mock.__class__.__name__} model answers:\n{model_answers}"
    )

    resp_prefix = test_results["mock_resp_prefix"]
    for prompt, model_answer in zip(prompts, model_answers):
        resp = f"{resp_prefix}{prompt[:common_consts.DEFAULT_MAX_TOKENS]}..."
        assert resp == model_answer
