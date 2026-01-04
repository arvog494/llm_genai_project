from __future__ import annotations
from typing import Any, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from .config import settings


class LLMClient:
    def __init__(
        self,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        temperature: float = 0.1,
    ):
        chat_model = chat_model or settings.ollama_chat_model
        embedding_model = embedding_model or settings.ollama_embedding_model

        chat_kwargs: dict[str, Any] = {
            "model": chat_model,
            "temperature": temperature,
            "base_url": settings.ollama_base_url,
        }
        embeddings_kwargs: dict[str, Any] = {
            "model": embedding_model,
            "base_url": settings.ollama_base_url,
        }

        # These knobs can reduce "gets worse over time" behavior by unloading models sooner
        # and/or reducing KV-cache growth (context + output).
        if settings.ollama_keep_alive is not None:
            chat_kwargs["keep_alive"] = settings.ollama_keep_alive
            embeddings_kwargs["keep_alive"] = settings.ollama_keep_alive
        if settings.ollama_num_ctx is not None:
            chat_kwargs["num_ctx"] = settings.ollama_num_ctx
        if settings.ollama_num_predict is not None:
            chat_kwargs["num_predict"] = settings.ollama_num_predict

        # Be defensive about langchain_ollama versions that may not support all params.
        try:
            self.chat = ChatOllama(**chat_kwargs)
        except TypeError:
            chat_kwargs.pop("keep_alive", None)
            chat_kwargs.pop("num_ctx", None)
            chat_kwargs.pop("num_predict", None)
            self.chat = ChatOllama(**chat_kwargs)

        try:
            self.embeddings = OllamaEmbeddings(**embeddings_kwargs)
        except TypeError:
            embeddings_kwargs.pop("keep_alive", None)
            self.embeddings = OllamaEmbeddings(**embeddings_kwargs)

    def simple_chat(self, system_prompt: str, user_prompt: str) :
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
            ]
        )
        chain = prompt | self.chat
        resp = chain.invoke({"input": user_prompt})
        return resp.content

    def embed_texts(self, texts: List[str]) :
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) :
        return self.embeddings.embed_query(text)
