from common.entities import Document, PromptingMode
from common.utils import get_messages_list
from src.wrappers import HfTokenizer

from typing import List, Tuple, Optional


class PromptBuilder:

    _TEMPLATE_MAPPING = {
        PromptingMode.OPENBOOK: {
            "system": "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).",
            "user": [
                "Search Results:\n{search_results}",
                "Question: {question}",
                "Answer:"
            ]
        },
        PromptingMode.OPENBOOK_RANDOM: {
            "system": "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). The search results are ordered randomly.",
            "user": [
                "Search Results:\n{search_results}",
                "Question: {question}",
                "Answer:"
            ]
        },
        PromptingMode.CLOSEDBOOK: {
            "system": "Write a high-quality answer for the given question.",
            "user": [
                "Question: {question}",
                "Answer:"
            ]
        }
    }

    def __init__(
        self,
        prompting_mode: PromptingMode,
        tokenizer: HfTokenizer
    ) -> None:

        self._tokenizer = tokenizer
        self._prompting_mode = prompting_mode
        self._system, self._user_template = self._get_prompt_components()

    def build(
        self,
        question: str,
        documents: Optional[List[Document]] = None
    ) -> str:

        if self._prompting_mode is PromptingMode.CLOSEDBOOK:
            user_prompt = self._user_template.format(question=question)
        else:
            search_results = self._format_documents(documents)
            user_prompt = self._user_template.format(
                search_results=search_results,
                question=question
            )

        messages = get_messages_list(user=user_prompt, system=self._system)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt

    def _get_prompt_components(self) -> Tuple[str, str]:
        prompt_template_parts = self._TEMPLATE_MAPPING[self._prompting_mode]
        syetem = prompt_template_parts["system"]
        user_parts = prompt_template_parts["user"]
        user_template = "\n\n".join(user_parts)
        return syetem, user_template

    def _format_documents(self, documents: List[Document]) -> str:
        return "\n".join(
            f"Document [{document_index}](Title: {document.title}) {document.text}"
            for document_index, document in enumerate(documents, 1)
        )
