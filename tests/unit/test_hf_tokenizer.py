from src.wrappers import HfTokenizer
from common.utils import get_messages_list
import common.consts as common_consts

from typing import Dict, Any
import pytest


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/hf_tokenizer.yaml"],
    indirect=True
)
def test_tokenizer_initialization(test_results: Dict[str, Any]) -> None:
    tokenizer_initialization = test_results["tokenizer_initialization"]
    for model in common_consts.SUPPORTED_MODELS:
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
    [common_consts.SUPPORTED_MODELS[0]],
    indirect=True
)
def test_apply_chat_template(
    hf_tokenizer: HfTokenizer,
    test_results: Dict[str, Any]
) -> None:

    prompt = test_results["prompt"]
    messages = get_messages_list(prompt)
    result = hf_tokenizer.apply_chat_template(messages)
    tokenized_messages_reference = test_results["tokenized_messages_reference"]

    assert len(result) == len(tokenized_messages_reference)
    assert all([
        res == ref
        for res, ref in zip(
            result, tokenized_messages_reference
        )
    ])


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/hf_tokenizer.yaml"],
    indirect=True
)
@pytest.mark.parametrize(
    "hf_tokenizer",
    [common_consts.SUPPORTED_MODELS[0]],
    indirect=True
)
def test_tokenize(
    hf_tokenizer: HfTokenizer,
    test_results: Dict[str, Any]
) -> None:

    prompt = test_results["prompt"]
    result = hf_tokenizer.tokenize(prompt)
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
    [common_consts.SUPPORTED_MODELS[0]],
    indirect=True
)
def test_count_tokens(
    hf_tokenizer: HfTokenizer,
    test_results: Dict[str, Any]
) -> None:

    prompt = test_results["prompt"]
    num_tokens = hf_tokenizer.count_tokens(prompt=prompt)
    tokenized_messages_reference = test_results["tokenized_messages_reference"]
    assert num_tokens == len(tokenized_messages_reference)
