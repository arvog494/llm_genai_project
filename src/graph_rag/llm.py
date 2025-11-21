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

        self.chat = ChatOllama(
            model=chat_model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=settings.ollama_base_url,
        )

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
