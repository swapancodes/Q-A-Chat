from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.retrieve import retrieve
from services.llm_service import llm

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.

Answer ONLY from the supplied context.

If the answer is unavailable, say:

I don't know.

Context:
{context}
"""
        ),
        ("human", "{question}")
    ]
)

chain = prompt | llm | StrOutputParser()


def ask_question(question):

    context = retrieve(question)

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
    return response