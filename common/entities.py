from pydantic.dataclasses import dataclass
from typing import TypeVar, Optional, Type
from copy import deepcopy
from enum import Enum

T = TypeVar("T")
from enum import Enum


class ExperimentType(Enum):
    TEST = "test"
    GOLD_IDX_CHANGE = "gold-idx-change"
    DOC_NUM_CHANGE = "doc-num-change"


class PromptingMode(Enum):
    CLOSEDBOOK = "closedbook"
    OPENBOOK = "openbook"
    OPENBOOK_RANDOM = "openbook-random"


# Adapted from the nelson-liu/lost-in-the-middle repository
# Original source: https://github.com/nelson-liu/lost-in-the-middle
# Licensed under the MIT License
# Modifications may have been made to suit specific project needs.
@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))
