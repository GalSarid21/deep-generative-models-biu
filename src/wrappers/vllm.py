from typing import Optional, List
from vllm import LLM, SamplingParams


class vLLMWrapper:
    """
    A wrapper class to abstract the vLLM package from the project.
    """

    def __init__(
        self,
        model: str,
        dtype: str,
        num_gpus: int,
        max_model_len: int,
        gpu_memory_utilization: float
    ) -> None:

        self._llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization
        )

    @property
    def llm(self) -> LLM:
        return self._llm

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> List[str]:

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        results = self._llm.generate(prompts, sampling_params)
        return [res.outputs[0].text for res in results]

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> str:

        return self.generate_batch(
            prompts=[prompt],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )[0]
