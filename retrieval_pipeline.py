from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os, sys

def get_embedding_model():
    return OllamaEmbeddings(model="mxbai-embed-large")

def main():
    # Start ingestion pipeline
    print("Starting RAG Retrieval pipeline")
    persist_direcotry = "db/chroma_db"
    
    query = "Tell me about Roadster"
    
    # Embed query and pull the chunks
    
    # Step 1: Load embedding model
    embedding_model = get_embedding_model()
    
    # Step 2 : Load vector store
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_direcotry,
        collection_configuration={"hnsw:space": "cosine"}
    )
    
    # Step 3: Configure retriever to load chunks
    vector_store_retriver = vector_store.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k" : 3, "score_threshold" : 0.5}
    )
    
    # Step 4 : Load chunks
    chunks = vector_store_retriver.invoke(query)
    
    # Step 5 : Print documents
    for i, doc in enumerate(chunks):
        print("="*100)
        print(f"{i+1}. {doc.metadata['source']}")
        print(f"{doc.page_content}\n")


if __name__ == "__main__":
    main()