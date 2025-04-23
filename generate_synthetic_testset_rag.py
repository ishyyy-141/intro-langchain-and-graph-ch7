import asyncio
import numpy as np
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

load_dotenv()
OAI_API_KEY = os.getenv("OAI_API_KEY")


class RAG:
    def __init__(
        self,
        oai_api_key,
        chat_model="gpt-4o",
        embeddings_model="text-embedding-3-small",
    ):
        self.llm = ChatOpenAI(model=chat_model, api_key=oai_api_key)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model, api_key=oai_api_key)
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            (
                "system",
                "You are a helpful assistant that answers questions based on given documents only.",
            ),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content


async def async_main():
    path = "Sample_Docs_Markdown/"
    loader = DirectoryLoader(path=path, glob="**/*.md")
    docs = loader.load()

    chat_model = ChatOpenAI(model="gpt-4.1-nano", api_key=OAI_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OAI_API_KEY
    )
    generator_llm = LangchainLLMWrapper(chat_model)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings
    )
    dataset = generator.generate_with_langchain_docs(documents=docs, testset_size=10)
    dataset.upload()


if __name__ == "__main__":
    asyncio.run(async_main())
