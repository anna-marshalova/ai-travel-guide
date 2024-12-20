import os

from dotenv import load_dotenv

from langchain_community.chat_models import GigaChat

from src.data.data_processing import load_and_preprocess_data

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.retriever import HierarchicalRetriever

load_dotenv()

giga_key = os.getenv("API_KEY")


class RAG:
    def __init__(self, data, model_name="GigaChat"):
        self.retriever = HierarchicalRetriever(data)
        self.llm = GigaChat(
            credentials=giga_key,
            model=model_name,
            timeout=30,
            verify_ssl_certs=False,
            profanity_check=False,
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Ты - опытный туристический гид с обширными знаниями о путешествиях, культуре и истории разных мест.
            Контекст из надежного источника: {context}

            Вопрос пользователя: {question}

            При ответе:
            1. В первую очередь используй факты из предоставленного контекста - это самая актуальная и проверенная информация
            2. Дополняй ответ релевантными общими знаниями о:
            - Культурных особенностях и традициях
            - Исторических фактах
            - Практических советах по путешествиям
            - Современных тенденциях туризма
            3. Четко разграничивай информацию из контекста и общие знания
            4. Структурируй информацию в удобном для чтения формате
            5. Если информация из разных источников противоречит друг другу, отдавай приоритет контексту
            6. Если в контексте недостаточно информации по какому-то аспекту вопроса - используй свои знания, но укажи это.

            Стремись дать максимально полезный, информативный и практичный ответ, комбинируя все доступные знания.
            
            Если вопрос пользователя не предполагает поиска, например пользователь написал просто "Привет!", можешь не опираться на информацию в контексте.

            Ответ:"""
        )
        self.rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
        )

    def retrieve(self, query):
        return self.retriever.retrieve(query)

    def run(self, query):
        context = self.retrieve(query)
        result = self.rag_chain.invoke({"question": query, "context": context})
        return {"response": result.content, "retrieved_chunks": context}


if __name__ == "__main__":
    chunked_data = load_and_preprocess_data(datadir="./data")

    rag = RAG(chunked_data)
    print(rag.run("Что посмотреть в Шанхае?"))
