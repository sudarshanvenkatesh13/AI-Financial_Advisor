from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 — Load the existing ChromaDB
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

# Step 2 — Format retrieved documents into a string
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Step 3 — Build the RAG chain using LCEL
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
    You are a personal finance assistant. Use the bank statement data below to answer the question.
    Only use the provided data — do not make things up.

    Bank Statement Data:
    {context}

    Question: {question}

    Answer clearly and helpfully:
    """)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # LCEL chain — this is the modern LangChain v1.x way
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Step 4 — Ask a question
def ask(question):
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)
    return chain.invoke(question)

# Step 5 — Test it!
if __name__ == "__main__":
    print("Question 1: How much did I spend on food?")
    print(ask("How much did I spend on food?"))
    print("---")
    print("Question 2: What is my biggest expense category?")
    print(ask("What is my biggest expense category?"))