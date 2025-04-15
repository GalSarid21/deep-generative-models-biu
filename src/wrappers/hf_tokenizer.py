from transformers import AutoTokenizer, TensorType
from typing import List, Dict, Any


class HfTokenizer:

    def __init__(self, model: str) -> None:
        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        **tokenizer_kwargs: Any
    ) -> (str | List[int]):
        """
        Exposing the 'apply_chat_template' method with its HF signature.
        """
        return self._tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            **tokenizer_kwargs
        )

    def count_tokens(self, **kwargs) -> int:
        """
        Function to count tokens using the 'apply_chat_template' method,
        meaning that we consider the prompt special tokens being added
        by the tokenizer in our count.

        Gets **kwargs input to handle both messages (List[Dict[str, str]])
        and prompt (str) inputs correctly.

        Raises an exception if none of those two variables are being passed.
        """
        if kwargs.get("prompt"):
            prompt = kwargs.get("prompt")
            messages = [{"role": "user", "content": prompt}]
        
        elif kwargs.get("messages"):
            messages = kwargs.get("messages")
        
        else:
            raise Exception(
                "'count_tokens' function must get either 'prompt' or " +
                "'messages' input variables."
            )

        tokenized_messages = self.apply_chat_template(messages)
        return len(tokenized_messages)
