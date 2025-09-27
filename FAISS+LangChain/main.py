from faiss_rag import rag_faiss_query

@app.post("/rag-faiss-query") 
def query_faiss_rag(request: QueryRequest):
    return rag_faiss_query(request.question)