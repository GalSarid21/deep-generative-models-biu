DATA_SRC_DIR = "./lost-in-the-middle/qa_data"
DATA_DST_DIR = "./qa_data"

SUPPORTED_NUM_DOCS = [10, 20, 30]
SUPPORTED_MODELS = [
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

INVALID_ENUM_CREATION_MSG = "Tried to create {obj} instance with invalid " \
    + "`prompting_mode` value: {arg}"
