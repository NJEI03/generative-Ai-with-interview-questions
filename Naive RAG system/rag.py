from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# 1. Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create your knowledge base (simple example)
documents = [
    "Paul Biya has been the president of Cameroon since 1982.",
    "FastAPI is a modern Python web framework for building APIs.",
    "The capital of France is Paris."
]

# 3. Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = [Document(page_content=text) for text in documents]
texts = text_splitter.split_documents(docs)

# 4. Create vector store
vector_store = Chroma.from_documents(texts, embeddings)

# 5. Create retriever
retriever = vector_store.as_retriever()

def rag_query(question: str):
    # 6. Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(question)
    
    # 7. Simple RAG: use the first relevant doc as context
    context = relevant_docs[0].page_content if relevant_docs else "No relevant information found."
    
    return {
        "question": question,
        "context": context,
        "answer": f"Based on the context: {context}"  # In next step, we'll use an LLM here
    }