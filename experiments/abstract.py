from typing import List, Any
from abc import ABC, abstractmethod


class AbstractExperiment(ABC):
    
    def __init__(self) -> None:
        pass

    def _load_experiment_data() -> List[Any]:
        pass
