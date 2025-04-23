import asyncio
import numpy as np
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

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
    chat_model = ChatOpenAI(model="gpt-4.1-nano", api_key=OAI_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OAI_API_KEY
    )
    evaluator_llm = LangchainLLMWrapper(chat_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    rag = RAG(oai_api_key=OAI_API_KEY)

    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    ]
    rag.load_documents(sample_docs)

    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?",
    ]
    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    ]
    dataset = []
    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query=query)
        response = rag.generate_answer(query=query, relevant_doc=relevant_docs)
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference,
            }
        )
    eval_dataset = EvaluationDataset.from_list(data=dataset)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
    )
    result.upload()


if __name__ == "__main__":
    asyncio.run(async_main())
