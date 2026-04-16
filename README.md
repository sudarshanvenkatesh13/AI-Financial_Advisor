# 💰 AI Personal Finance Advisor

Ask questions about your bank statement in plain English — powered by RAG + LangChain + OpenAI.

## 🎯 What it does

Upload your bank statement (CSV) and ask questions like:
- *"How much did I spend on food last month?"*
- *"What is my biggest expense category?"*
- *"How much did I spend on transport?"*

The AI reads YOUR data, finds the relevant transactions, and answers accurately — no hallucinations.

## 🏗️ How it works

```
CSV Upload → Pandas Parser → OpenAI Embeddings → ChromaDB (Vector DB)
                                                        ↓
User Question → Embedding → Semantic Search → LangChain → GPT-3.5 → Answer
```

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Vector database for semantic search |
| **OpenAI** | Embeddings + GPT-3.5-turbo |
| **FastAPI** | REST API backend |
| **Streamlit** | Interactive web UI |

## 🚀 Run it locally

**1. Clone the repo**
```bash
git clone https://github.com/sudarshanvenkatesh13/AI-Financial_Advisor.git
cd AI-Financial_Advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI key**
```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

**4. Run the web app**
```bash
python -m streamlit run app.py
```

**5. Or run the API instead**
```bash
uvicorn main:app --reload
# API docs at http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```
AI-Financial_Advisor/
├── app.py                  # Streamlit web UI
├── main.py                 # FastAPI backend
├── ingest.py               # CSV parsing + ChromaDB ingestion
├── query.py                # RAG chain + question answering
├── requirements.txt        # Dependencies
├── .env.example            # Environment variable template
└── sample_data/
    └── sample_bank.csv     # Sample data to try it out
```

## 📊 Sample CSV Format

Your CSV should have these columns:

```
Date,Description,Amount,Category
2024-01-01,Swiggy,-450,Food
2024-01-02,Salary,80000,Income
```

## 👤 Built by Sudarshan Venkatesh

[GitHub](https://github.com/sudarshanvenkatesh13) · [LinkedIn](https://linkedin.com/in/sudarshan2020) · [Twitter](https://x.com/SudarshanVenk)
