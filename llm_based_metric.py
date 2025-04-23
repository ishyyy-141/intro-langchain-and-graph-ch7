import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import SingleTurnSample
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

    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    }

    metric = AspectCritic(
        name="summary_accuracy",
        llm=evaluator_llm,
        definition="Verify if the summary is accurate.",
    )
    test_data = SingleTurnSample(**test_data)
    result = await metric.single_turn_ascore(test_data)
    print(result)


if __name__ == "__main__":
    asyncio.run(async_main())
