from typing import List

from insight_engine_core.database.vector_store import SimilaritySearchResult
from insight_engine_core.llm_interface.base_llm import BaseLLM
from ..rag.retriever import Retriever
from ..processing.embedder import Embedder # For type hints
from ..database.vector_store import VectorStore, SimilaritySearchResult # For type hints

class Generator:
    def __init__(self, retriever: Retriever, llm_interface: BaseLLM):
        self.retriever = retriever
        self.llm_interface = llm_interface

    def _format_context(self, context_docs: List[SimilaritySearchResult]) -> str:
        context_docs_str = "Context:\n"
        if len(context_docs) == 0:
            return ""

        context_parts = ["Context:"]
        for i, doc in enumerate(context_docs):
            # Use 1-based indexing for human-readable list
            context_parts.append(f"{i + 1}. {doc.chunk_text.strip()}")
        return "\n".join(context_parts)

    def _construct_prompt(self, query: str, formatted_context: str) -> str:
        if formatted_context != "":
            prompt = (f"{formatted_context}\n\nBased on the above context, answer the following question:\n"
                      f"Question: {query}")
        else:
            prompt = f"Answer the following question:\nQuestion: {query}"

        return prompt

    def generate_response(self, query_text: str, retriever_top_k: int = 5) -> str:
        retriever = self.retriever.retrieve(query_text, top_k=retriever_top_k)
        context = self._format_context(retriever)
        final_prompt_to_llm = self._construct_prompt(query=query_text, formatted_context=context)
        llm_response = self.llm_interface.generate(prompt=final_prompt_to_llm)
        return llm_response
