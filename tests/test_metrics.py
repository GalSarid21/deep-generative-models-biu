from src.metrics import best_subspan_em, normalize_sentence

from typing import Dict, Any
import logging
import pytest


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/metrics.yaml"],
    indirect=True
)
def test_best_subset_em_metric(test_results: Dict[str, Any]) -> None:

    logging.info(f"Testing best_subspan_em for 'good answer'")
    res = best_subspan_em(
        prediction=test_results["good_model_answer"],
        ground_truths=test_results["answers"]
    )
    assert res == 1.0
    
    logging.info(f"Testing best_subspan_em for 'bad answer'")
    res = best_subspan_em(
        prediction=test_results["bad_model_answer"],
        ground_truths=test_results["answers"]
    )
    assert res == 0.0


@pytest.mark.parametrize(
    "test_results",
    ["./tests/results/metrics.yaml"],
    indirect=True
)
def test_normalize_sentence(test_results: Dict[str, Any]) -> None:
    model_answer = test_results["bad_model_answer"]
    normalized_model_answer = normalize_sentence(model_answer)
    assert normalized_model_answer == test_results["normalized_model_answer"]
