from common.entities import ExperimentType, PromptingMode
import common.consts as consts

from argparse import ArgumentParser, Namespace
import logging
import json


def read_cli_env_args() -> Namespace:
    parser = ArgumentParser("")

    parser.add_argument(
        "--experiment",
        help="experiment type to run.",
        type=str,
        choices=[et.value for et in ExperimentType],
        default=ExperimentType.GOLD_IDX_CHANGE
    )

    parser.add_argument(
        "--hf_token",
        help="HF Token for model downloading.",
        type=str
    )

    parser.add_argument(
        "--model",
        help="HF model repo to use in experiment.",
        type=str,
        choices=consts.SUPPORTED_MODELS,
        default=consts.SUPPORTED_MODELS[0] # "tiiuae/Falcon3-Mamba-7B-Instruct"
    )

    parser.add_argument(
        "--dtype",
        help="torch dtype to use during model loading.",
        type=str,
        choices=consts.SUPPORTED_DTYPES,
        default=consts.SUPPORTED_DTYPES[0] # bfloat16
    )

    parser.add_argument(
        "--prompting_mode",
        help="prompting type to use in experiment.",
        type=str,
        choices=[pm.value for pm in PromptingMode],
        default=PromptingMode.OPENBOOK_RANDOM.value
    )

    parser.add_argument(
        "--num_gpus",
        help="number of GPUs to use in experiment.",
        type=int,
        default=consts.DEFAULT_NUM_GPUS
    )

    parser.add_argument(
        "--temperature",
        help="temperature to use in generation.",
        type=float,
        default=consts.DEFAULT_TEMPERATURE
    )

    parser.add_argument(
        "--top_p",
        help="top-p to use in generation.",
        type=float,
        default=consts.DEFAULT_TOP_P
    )

    parser.add_argument(
        "--max_tokens",
        help="maximum number of new tokens to generate.",
        type=int,
        default=consts.DEFAULT_MAX_TOKENS,
    )

    # relevant for gold_idx_change experiment
    parser.add_argument(
        "--num_docs",
        help="number of documents to use in the experiment [when relevant].",
        type=int,
        choices=consts.SUPPORTED_NUM_DOCS,
        default=consts.SUPPORTED_NUM_DOCS[0]
    )

    # relevant for num_docs_change experiment
    parser.add_argument(
        "--golden_idx",
        help="index of golden answer index to use in the experiment [when relevant].",
        type=int,
        choices=consts.SUPPORTED_GOLD_IDXS,
        default=consts.SUPPORTED_GOLD_IDXS[0]
    )

    parser.add_argument(
        "--test_mode",
        help="boolean that indicates the experiment should run in test mode.",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--results_dir",
        help="external directory to store the experiment results.",
        type=str
    )

    parser.add_argument(
        "--max_model_len",
        help="maximum tokens the model can get as input.",
        type=int,
        default=consts.DEFAULT_MAX_MODEL_LEN
    )

    args = parser.parse_args()
    logging.info(
        "Environment Variables:\n" +
        json.dumps(vars(args), indent=2)
    )
    return args
