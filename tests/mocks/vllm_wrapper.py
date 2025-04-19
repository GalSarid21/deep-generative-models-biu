from unittest.mock import MagicMock
from typing import List, Optional
import logging
import json


class vLLMWrapperMock:

    def __init__(
        self,
        model: str,
        dtype: str,
        num_gpus: int,
        max_model_len: int,
        gpu_memory_utilization: float
    ) -> None:
        # log input parameters
        params = locals()
        all_params_dict = {
            key: value
            for key, value in params.items()
            if key != "self"
        }
        logging.info(
            f"{self.__class__.__name__} input params:\n" +
            json.dumps(all_params_dict, indent=2)
        )
        self._llm = MagicMock()

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> List[str]:
        # simulates the generate_batch method response with MagicMock
        mocked_results = [
            MagicMock(
                outputs=[
                    MagicMock(
                        text=f"Mocked response to: {prompt[:max_tokens]}..."
                    )
            ]) for prompt in prompts
        ]
        self._llm.generate_batch.return_value = mocked_results
        results = self._llm.generate_batch(
            prompts, temperature, max_tokens, top_p
        )
        return [res.outputs[0].text for res in results]

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> str:
        # use generate_batch internally like in the original vLLMWrapper
        return self.generate_batch([prompt], temperature, max_tokens, top_p)[0]
