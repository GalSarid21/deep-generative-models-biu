from src.wrappers.hf_tokenizer import HfTokenizer
import common.consts as consts

from typing import Dict, Any
import pytest


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/hf_tokenizer.yaml"],
    indirect=True
)
def test_tokenizer_initialization(test_results: Dict[str, Any]) -> None:
    tokenizer_initialization = test_results["tokenizer_initialization"]
    for model in consts.SUPPORTED_MODELS:
        tokenizer = HfTokenizer(model)
        assert tokenizer.eos_token_id == tokenizer_initialization[model]["eos_token_id"]
        assert tokenizer.eos_token == tokenizer_initialization[model]["eos_token"]


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/hf_tokenizer.yaml"],
    indirect=True
)
@pytest.mark.parametrize(
    "hf_tokenizer",
    [consts.SUPPORTED_MODELS[0]],
    indirect=True
)
def test_apply_chat_template(
    hf_tokenizer: HfTokenizer,
    test_results: Dict[str, Any]
) -> None:

    prompt = test_results["prompt"]
    messages = [{"role": "user", "content": prompt}]
    result = hf_tokenizer.apply_chat_template(messages)
    tokenized_reference = test_results["tokenized_reference"]

    assert len(result) == len(tokenized_reference)
    assert all([res == ref for res, ref in zip(result, tokenized_reference)])


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/hf_tokenizer.yaml"],
    indirect=True
)
@pytest.mark.parametrize(
    "hf_tokenizer",
    [consts.SUPPORTED_MODELS[0]],
    indirect=True
)
def test_count_tokens(
    hf_tokenizer: HfTokenizer,
    test_results: Dict[str, Any]
) -> None:

    prompt = test_results["prompt"]
    num_tokens = hf_tokenizer.count_tokens(prompt=prompt)
    assert num_tokens == test_results["prompt_num_tokens"]
