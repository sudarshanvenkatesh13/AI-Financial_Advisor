from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import pandas as pd
import io

load_dotenv()

app = FastAPI(title="AI Finance Advisor")

# Allow all origins (so we can test from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: build vectorstore ────────────────────────────────
def get_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings()
    )

# ── Helper: format docs ──────────────────────────────────────
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ── Helper: build RAG chain ──────────────────────────────────
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

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Route 1: Health check ────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI Finance Advisor is running!"}

# ── Route 2: Upload CSV bank statement ───────────────────────
@app.post("/upload")
async def upload_statement(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Convert each row to a Document
    documents = []
    for _, row in df.iterrows():
        text = f"On {row['Date']}, {row['Description']} — Amount: ₹{row['Amount']} — Category: {row['Category']}"
        documents.append(Document(page_content=text))

    # Store in ChromaDB
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return {
        "message": f"Successfully uploaded and stored {len(documents)} transactions!"
    }

# ── Route 3: Ask a question ───────────────────────────────────
@app.post("/query")
def query_statement(payload: dict):
    question = payload.get("question", "")
    if not question:
        return {"error": "No question provided"}

    vectorstore = get_vectorstore()
    chain = build_chain(vectorstore)
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer
    }