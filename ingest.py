import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os 

# set up the OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# step 2 - load the csv bank statements
def load_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} transactions")
    return df

# step 3 - convert each row into a langcahin document 

def df_to_documents(df):
    documents = []
    for _, row in df.iterrows():
        text = f"On {row['Date']}, {row['Description']} — Amount: ₹{row['Amount']} — Category: {row['Category']}"
        doc = Document(page_content=text)
        documents.append(doc)
    return documents

#step 4 - embed and store in chromaDB

def ingest_tochromadb(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(documents)} documents in ChromaDB")
    return vectorstore

#step5 - run everything 
if __name__ == "__main__":
    df = load_csv("sample_data/sample_bank.csv")
    documents = df_to_documents(df)
    ingest_tochromadb(documents)