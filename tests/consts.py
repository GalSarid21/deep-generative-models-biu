DOCUMENTS_FOLDER_PATH = "./qa_data/10_total_documents"
DOCUMENTS_FOLDER_TEMPLATE = "./qa_data/{num_docs}_total_documents"
DOCUMENT_NAME_TEMPLATE = "nq-open-{num_docs}_total_documents_gold_at_{gold_idx}.jsonl"

DOCUMETS_FOLDER_FILES = [
    "nq-open-10_total_documents_gold_at_0.jsonl",
    "nq-open-10_total_documents_gold_at_4.jsonl",
    "nq-open-10_total_documents_gold_at_9.jsonl"
]

TEST_MODELS = [
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

TEST_DOCUMENT_PATH = f"{DOCUMENTS_FOLDER_PATH}/{DOCUMETS_FOLDER_FILES[0]}"
NUM_DOCS_TO_TEST = 2