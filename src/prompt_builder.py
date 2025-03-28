from common.entities import Document, PromptingMode

from typing import List, Optional


class PromptBuilder:

    _OPENBOOK_PROMPT_TEMPLATE_PARTS = [
        "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).",
        "Search Results:\n{search_results}",
        "Question: {question}",
        "Answer:"
    ]

    _OPENBOOK_RANDOM_PROMPT_TEMPLATE_PARTS = [
        "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). The search results are ordered randomly.",
        "Search Results:\n{search_results}",
        "Question: {question}",
        "Answer:"
    ]

    _CLOSED_BOOK_PROMPT_TEMPLATE_PARTS = [
        "Write a high-quality answer for the given question."
        "Question: {question}",
        "Answer:"
    ]

    def __init__(self, prompting_mode: str) -> None:
        self._pompting_mode = PromptingMode(prompting_mode)
        self._prompt_template = self._get_prompt_template()

    def build(
        self,
        question: str,
        documents: List[Document]
    ) -> str:

        formatted_documents = [
            f"Document [{document_index}](Title: {document.title}) {document.text}"
            for document_index, document in enumerate(documents, 1)
        ]

        search_results = "\n".join(formatted_documents)
        return self._prompt_template.format(
            search_results=search_results,
            question=question
        )

    def _get_prompt_template(self) -> str:

        prompt_template_parts_mappings = {
            PromptingMode.CLOSEDBOOK: self._CLOSED_BOOK_PROMPT_TEMPLATE_PARTS,
            PromptingMode.OPENBOOK: self._OPENBOOK_PROMPT_TEMPLATE_PARTS,
            PromptingMode.OPENBOOK_RANDOM: self._OPENBOOK_RANDOM_PROMPT_TEMPLATE_PARTS
        }

        prompt_template_parts = prompt_template_parts_mappings[self._pompting_mode]
        prompt_template = "\n\n".join(prompt_template_parts)
        return prompt_template
