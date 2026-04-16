# 💰 AI Personal Finance Advisor

Ask questions about your spending in plain English — powered by RAG + LangChain + OpenAI.

## 🎯 What it does
Upload your bank statement (CSV) and ask questions like:
- "How much did I spend on food last month?"
- "What is my biggest expense category?"
- "Am I spending more than I earn?"

The AI retrieves relevant transactions from your data and answers accurately — no hallucinations.

## 🏗️ Architecture
CSV Upload → Pandas Parser → OpenAI Embeddings → ChromaDB (Vector DB)
↓
User Question → Embedding → Semantic Search → LangChain LCEL → GPT-3.5 → Answer

## 🛠️ Tech Stack
- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — Vector database for semantic search
- **OpenAI** — Embeddings + GPT-3.5-turbo
- **FastAPI** — REST API backend
- **Streamlit** — Interactive web UI

## 🚀 Run it locally

1. Clone the repo
```bash
git clone https://github.com/sudarshanvenkatesh13/ai-finance-advisor
cd ai-finance-advisor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your OpenAI key
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

4. Run the Streamlit app
```bash
python -m streamlit run app.py
```

## 📁 Project Structure
finance-advisor/
├── app.py              # Streamlit web UI
├── main.py             # FastAPI backend
├── ingest.py           # CSV parsing + ChromaDB ingestion
├── query.py            # RAG chain + question answering
├── requirements.txt
├── sample_data/
│   └── sample_bank.csv # Sample data to try it out
└── .env.example

## 👤 Built by
Sudarshan Venkatesh — [GitHub](https://github.com/sudarshanvenkatesh13) | 
[LinkedIn](https://linkedin.com/in/sudarshan2020) | [Twitter](https://x.com/SudarshanVenk)