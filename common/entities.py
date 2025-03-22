from enum import Enum


class ExperimentType(Enum):
    TEST = "test"
    GOLD_IDX_CHANGE = "gold-idx-change"
    DOC_NUM_CHANGE = "doc-num-change"
