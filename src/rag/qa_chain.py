"""
LangChain QA Chain Module (LCEL)

Constructs RAG chain synthesizing natural-language answers with explicit timestamp citations.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from src.rag.retriever import EventRetriever


PROMPT_TEMPLATE = """You are an expert soccer match analyst AI assistant.
Answer the user's question accurately using ONLY the provided soccer match event logs below.

For every event or situation you reference, YOU MUST include the exact timestamp window formatted as [MM:SS - MM:SS].

Retrieved Match Event Logs:
{context}

User Question: {question}

Detailed Tactical Answer (including exact timestamps):"""


class SoccerQAChain:
    """
    LangChain QA Chain using LCEL.
    """

    def __init__(self, retriever: EventRetriever, provider: str = "local", model_name: str = "default"):
        self.retriever = retriever
        self.provider = provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self._build_chain()

    def _initialize_llm(self) -> Runnable:
        """Initializes swappable LLM backend based on provider configuration."""
        if self.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model_name=self.model_name, temperature=0.1)
            except Exception as e:
                print(f"[!] OpenAI initialization failed: {e}. Falling back to local model.")

        elif self.provider == "ollama":
            try:
                from langchain_community.llms import Ollama
                return Ollama(model=self.model_name)
            except Exception as e:
                print(f"[!] Ollama initialization failed: {e}. Falling back to local model.")

        # Default local lightweight synthesis LLM / Fallback explicitly wrapped as RunnableLambda
        analyst = LocalMatchAnalystLLM()
        return RunnableLambda(analyst.invoke)

    def _build_chain(self):
        """Builds LCEL RAG chain."""
        return (
            {"context": self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents for prompt context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(formatted)

    def answer_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Executes QA chain for natural language query.
        Returns (synthesized_answer, list_of_retrieved_docs).
        """
        docs = self.retriever.retrieve(question)

        if not docs:
            return "No relevant soccer match events were found for your query.", []

        # Generate answer with LCEL chain
        answer = self.chain.invoke(question)

        return answer, docs


class LocalMatchAnalystLLM(Runnable):
    """
    Lightweight deterministic local LLM synthesizer for offline demo execution.
    Subclasses Runnable and implements __call__ for full LCEL pipeline compatibility.
    """

    def __call__(self, input: Any, config: Optional[Any] = None) -> str:
        return self.invoke(input, config)

    def invoke(self, input: Any, config: Optional[Any] = None) -> str:
        prompt_text = str(input.to_string()) if hasattr(input, "to_string") else str(input)

        # Extract context block from prompt
        if "Retrieved Match Event Logs:" in prompt_text:
            context_block = prompt_text.split("Retrieved Match Event Logs:")[1].split("User Question:")[0].strip()
        else:
            context_block = prompt_text

        lines = [line.strip() for line in context_block.split("\n") if line.strip() and line.strip().startswith("[")]

        if not lines:
            return "Based on the match logs, no matching event timestamps were found."

        summary_bullets = []
        for l in lines:
            summary_bullets.append(f"- {l}")

        response = (
            f"Based on the match analysis tracking logs, here are the relevant match events:\n\n"
            + "\n".join(summary_bullets)
            + "\n\nRefer to the embedded video player above scrubbed to these exact timestamp ranges."
        )
        return response
