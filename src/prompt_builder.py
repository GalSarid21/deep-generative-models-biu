from common.entities import Document, PromptingMode

from typing import List


class PromptBuilder:

    _TEMPLATE_MAPPING = {
        PromptingMode.OPENBOOK: [
            "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).",
            "Search Results:\n{search_results}",
            "Question: {question}",
            "Answer:"
        ],
        PromptingMode.OPENBOOK_RANDOM: [
            "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). The search results are ordered randomly.",
            "Search Results:\n{search_results}",
            "Question: {question}",
            "Answer:"
        ],
        PromptingMode.CLOSEDBOOK: [
            "Write a high-quality answer for the given question.",
            "Question: {question}",
            "Answer:"
        ]
    }

    def __init__(self, prompting_mode: PromptingMode) -> None:
        self._prompting_mode = prompting_mode
        self._prompt_template = self._get_prompt_template()

    def build(
        self,
        question: str,
        documents: List[Document]
    ) -> str:

        search_results = self._format_documents(documents)
        return self._prompt_template.format(
            search_results=search_results,
            question=question
        )

    def _get_prompt_template(self) -> str:
        prompt_template_parts = self._TEMPLATE_MAPPING[self._prompting_mode]
        prompt_template = "\n\n".join(prompt_template_parts)
        return prompt_template

    def _format_documents(self, documents: List[Document]) -> str:
        return "\n".join(
            f"Document [{document_index}](Title: {document.title}) {document.text}"
            for document_index, document in enumerate(documents, 1)
        )
