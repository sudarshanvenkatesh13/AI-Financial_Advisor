import streamlit as st
import pandas as pd
import io
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

# Works both locally and on Streamlit Cloud
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Finance Advisor",
    page_icon="💰",
    layout="centered"
)

# ── Title ────────────────────────────────────────────────────
st.title("💰 AI Finance Advisor")
st.markdown("Upload your bank statement and ask questions about your spending — powered by AI.")
st.divider()

# ── Helper functions ─────────────────────────────────────────
def ingest_csv(df):
    documents = []
    for _, row in df.iterrows():
        text = f"On {row['Date']}, {row['Description']} — Amount: ₹{row['Amount']} — Category: {row['Category']}"
        documents.append(Document(page_content=text))

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

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

# ── Section 1: Upload ─────────────────────────────────────────
st.subheader("📁 Step 1 — Upload your bank statement")
st.markdown("Don't have one? Download our [sample CSV](sample_data/sample_bank.csv) to try it out!")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} transactions!")
    st.dataframe(df, use_container_width=True)

    with st.spinner("🔄 Processing and storing in vector database..."):
        vectorstore = ingest_csv(df)
        st.session_state["vectorstore"] = vectorstore

    st.success("✅ Bank statement ingested into AI memory!")

st.divider()

# ── Section 2: Ask ────────────────────────────────────────────
st.subheader("💬 Step 2 — Ask anything about your spending")

question = st.text_input(
    "Your question",
    placeholder="e.g. How much did I spend on food? What is my biggest expense?"
)

if st.button("🤖 Ask AI", type="primary"):
    if "vectorstore" not in st.session_state:
        st.error("⚠️ Please upload a bank statement first!")
    elif not question:
        st.error("⚠️ Please type a question!")
    else:
        with st.spinner("🤔 Thinking..."):
            chain = build_chain(st.session_state["vectorstore"])
            answer = chain.invoke(question)

        st.divider()
        st.subheader("🤖 AI Answer")
        st.markdown(f"**Q: {question}**")
        st.info(answer)

st.divider()
st.caption("Built with LangChain + ChromaDB + OpenAI + Streamlit | by Sudarshan Venkatesh")