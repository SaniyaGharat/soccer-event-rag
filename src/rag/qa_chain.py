"""
LangChain QA Chain Module (LCEL)

Constructs RAG chain synthesizing natural-language answers with explicit timestamp citations.
Supports swappable LLM providers (Local, HuggingFace, Ollama, OpenAI) with runtime fault-tolerance.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
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
    LangChain QA Chain using LCEL with swappable LLM backends & runtime fallback.
    """

    def __init__(self, retriever: EventRetriever, provider: str = "local", model_name: str = "default"):
        self.retriever = retriever
        self.provider = provider
        self.model_name = model_name
        self.active_provider_name = "Local Match Analyst (Offline)"
        self.llm = self._initialize_llm()
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self._build_chain()

    def _initialize_llm(self) -> Runnable:
        """Initializes swappable LLM backend based on provider configuration."""
        if self.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                self.active_provider_name = f"OpenAI ({self.model_name if self.model_name != 'default' else 'gpt-3.5-turbo'})"
                return ChatOpenAI(model_name=self.model_name if self.model_name != "default" else "gpt-3.5-turbo", temperature=0.1)
            except Exception as e:
                print(f"[!] OpenAI initialization failed: {e}. Falling back to local model.")

        elif self.provider == "ollama":
            try:
                from langchain_community.llms import Ollama
                self.active_provider_name = f"Ollama ({self.model_name if self.model_name != 'default' else 'llama3'})"
                return Ollama(model=self.model_name if self.model_name != "default" else "llama3")
            except Exception as e:
                print(f"[!] Ollama initialization failed: {e}. Falling back to local model.")

        elif self.provider == "huggingface":
            try:
                from langchain_community.llms import HuggingFacePipeline
                model_id = self.model_name if self.model_name != "default" else "google/flan-t5-base"
                self.active_provider_name = f"HuggingFace ({model_id})"
                hf_llm = HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 256, "temperature": 0.1}
                )
                return hf_llm
            except Exception as e:
                print(f"[!] HuggingFace Pipeline initialization failed: {e}. Falling back to local model.")

        # Default local lightweight synthesis LLM / Fallback explicitly wrapped as RunnableLambda
        self.active_provider_name = "Local Match Analyst (Offline)"
        analyst = LocalMatchAnalystLLM()
        return RunnableLambda(analyst.invoke)

    def _build_chain(self):
        """Builds LCEL RAG chain."""
        return (
            self.prompt
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
        Executes QA chain for natural language query with runtime fault tolerance.
        Returns (synthesized_answer, list_of_retrieved_docs).
        """
        docs = self.retriever.retrieve(question)

        if not docs:
            return "No relevant soccer match events were found for your query.", []

        formatted_context = self._format_docs(docs)

        try:
            answer = self.chain.invoke({"context": formatted_context, "question": question})
        except Exception as e:
            err_msg = str(e)
            print(f"[!] Primary LLM provider ({self.provider}) execution failed: {err_msg}. Falling back to Local Synthesizer.")
            self.active_provider_name = f"{self.provider.upper()} (Unavailable: {err_msg[:40]}...) ➔ Fallback: Local Analyst"
            fallback_llm = LocalMatchAnalystLLM()
            prompt_str = self.prompt.format(context=formatted_context, question=question)
            answer = fallback_llm.invoke(prompt_str)

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
