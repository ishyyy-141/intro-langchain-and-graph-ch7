import asyncio
import os

from datasets import load_dataset

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.metrics import AspectCritic
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()


async def async_main():
    OAI_API_KEY = os.getenv("OAI_API_KEY")
    chat_model = ChatOpenAI(model="gpt-4.1-nano", api_key=OAI_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OAI_API_KEY
    )
    evaluator_llm = LangchainLLMWrapper(chat_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    metric = AspectCritic(
        name="summary_accuracy",
        llm=evaluator_llm,
        definition="Verify if the summary is accurate.",
    )

    eval_dataset = load_dataset(
        path="explodinggradients/earning_report_summary", split="train"
    )
    eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)

    result = evaluate(dataset=eval_dataset, metrics=[metric])
    result.upload()


if __name__ == "__main__":
    asyncio.run(async_main())
