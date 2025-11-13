# AI Developer Task: RAG-based Chatbot (Company Policies)

This project is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions about company policies. It fulfills all task requirements, including ingesting PDF documents, using a RAG pipeline, providing citations, and implementing conversational memory.

## 🚀 Features

* **PDF Ingestion:** Automatically ingests and processes all PDF files from a `docs` folder.
* **RAG Pipeline:** Uses a RAG (Retrieval-Augmented Generation) chain to find the most relevant policy information.
* **No Hallucination:** The LLM is strictly instructed via a custom prompt to answer *only* based on the retrieved context.
* **Citations:** Every answer is accompanied by the source document and page number it was drawn from.
* **Conversational Memory (Bonus):** The chatbot understands follow-up questions (e.g., "What about for part-time employees?") by maintaining chat history.

## 🛠️ Tech Stack

* **Framework:** LangChain
* **Interface:** Streamlit
* **LLM:** Google Gemini (google/flan-t5-large)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
* **Vector Store:** FAISS (Facebook AI Similarity Search)

## 📸 Visualizations
<img width="1915" height="971" alt="Screenshot 2025-11-14 003733" src="https://github.com/user-attachments/assets/875e0902-1347-40c7-b62f-40500c4c9144" />
<img width="1915" height="964" alt="Screenshot 2025-11-14 001213" src="https://github.com/user-attachments/assets/a74a2031-1416-4369-8673-f88adc15ae8b" />
<img width="802" height="342" alt="Screenshot 2025-11-14 000700" src="https://github.com/user-attachments/assets/936ddd0b-13c2-45f1-8053-134b6f265c96" />
<img width="649" height="269" alt="Screenshot 2025-11-13 223910" src="https://github.com/user-attachments/assets/41de36cf-5745-4c5a-92e5-d2f39ff44026" />


## ⚙️ Setup & Running Instructions

Follow these steps to run the project locally.

## 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd ai_chatbot_task
```


## 2. Install Dependencies

### Create a virtual environment
```bash
python -m venv venv
```
### Activate it (Windows)
```bash
.\venv\Scripts\activate
```
### Install required libraries
```bash
pip install -r requirements.txt
```

## 3. Set API Keys
### Add your key to the .env file:

### From Hugging
```bash
HUGGINGFAC_API_KEY="YOUR_API_KEY_HERE"
```
## 4. Add Policy Documents
Place your company's policy PDFs (or the included sample PDFs) into the /data folder.

## 5. Run the RAG_ENGINE.PY Script
You only need to run this script once to index your documents. It reads the PDFs from /data and creates a vectorstore folder.
```bash
python rag_engine.py
```


## 6. Run the Chatbot App (app.py file)
Start the Streamlit application.

```bash
streamlit run app.py
```

