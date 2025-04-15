RESULTS_DIR = "./results"
DATA_SRC_DIR = "./lost-in-the-middle/qa_data"
DATA_DST_DIR = "./qa_data"
EXPERIMENT_DATA_FOLDER = "./qa_data/{num_docs}_total_documents"

SUPPORTED_NUM_DOCS = [10, 20, 30]
SUPPORTED_MODELS = [
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

SUPPORTED_DTYPES = ["bfloat16", "float16"]
DEFAULT_MAX_TOKENS = 100
DEFAULT_NUM_GPUS = 1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0