import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# The only LLM wrapper guaranteed to work
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Load environment variables if needed
load_dotenv()

class RAGEngine:
    """
    RAG Engine using local HuggingFace models (NO OpenAI, NO HuggingFaceHub objects).
    """

    def __init__(self, data_dir: str = "data", vectorstore_dir: str = "vectorstore"):
        self.data_dir = data_dir
        self.vectorstore_dir = vectorstore_dir

        # Embeddings: No API, never fails
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # LLM: Use transformers pipeline, guaranteed to be available
        gen_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",  # You may swap for a smaller model if you lack RAM
            max_length=512
        )
        self.llm = HuggingFacePipeline(pipeline=gen_pipeline)

        self.vectorstore = None
        self.retriever = None

    def load_documents(self) -> List[Any]:
        print(f"📄 Loading documents from {self.data_dir}...")
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} document pages")
        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Any]):
        print("🔢 Creating vector store (this may take a moment)...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vectorstore_dir
        )
        print("✓ Vector store created successfully!")

    def load_existing_vectorstore(self):
        print("📂 Loading existing vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_dir,
            embedding_function=self.embeddings
        )
        print("✓ Vector store loaded successfully!")

    def setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("✓ Retriever configured")

    def generate_answer(self, question: str, history: str = "") -> Dict[str, Any]:
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant for company policies.

Context from documents:
{context}

{"Previous conversation:\n" + history if history else ""}

User question: {question}

Answer as clearly as possible, using ONLY the context above. If you do not know the answer, say "I don't have enough information in the company policies to answer that question." Always cite the document and page when answering.
"""
        answer = self.llm.invoke(prompt)
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        return {"answer": str(answer), "sources": sources}

    def initialize(self, force_rebuild: bool = False):
        print("\n🚀 Initializing RAG Engine...\n")
        vectorstore_exists = os.path.exists(self.vectorstore_dir) and os.listdir(self.vectorstore_dir)
        if force_rebuild or not vectorstore_exists:
            documents = self.load_documents()
            if not documents:
                raise ValueError(f"No PDF documents found in {self.data_dir} folder!")
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)
        else:
            self.load_existing_vectorstore()
        self.setup_retriever()
        print("\n✅ RAG Engine initialized successfully!\n")

    def query(self, question: str, history: str = "") -> Dict[str, Any]:
        return self.generate_answer(question, history)

    def reset_vectorstore(self):
        self.initialize(force_rebuild=True)

if __name__ == "__main__":
    print("Testing RAG Engine...\n")
    rag = RAGEngine()
    rag.initialize()
    result = rag.query("What is the company's leave policy?")
    #print(f"\n💬 Question: What is the company's leave policy?")
    print(f"\n📝 Answer:\n{result['answer']}")
    print(f"\n📚 Sources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source['source']} (Page {source['page']})")
