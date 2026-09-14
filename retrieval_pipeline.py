from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

import os, sys

def get_embedding_model():
    return OllamaEmbeddings(model="mxbai-embed-large")

def get_llm_model():
    return ChatOllama(model="llama3.1:8b")

def get_llm_response(documents, query):
    model = get_llm_model()
    
    llm_input = f"""Based on the following documents, please answer the question : {query}
    
    Documents : 
    {chr(10).join(f"- {doc.page_content}" for doc in documents)}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents.
    """
    messages = [
        SystemMessage(content="You are helpful assistant"),
        HumanMessage(content=llm_input)
    ]
    
    llm_response = model.invoke(messages)
    
    return llm_response.content

def main():
    # Start ingestion pipeline
    print("Starting RAG Retrieval pipeline")
    persist_direcotry = "db/chroma_db"
    
    query = "tell me about Roadster in brief"
    
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
        search_kwargs = {"k" : 3, "score_threshold" : 0.6}
    )
    
    # Step 4 : Load chunks
    chunks = vector_store_retriver.invoke(query)
    
    # Step 5 : Print documents
    # for i, doc in enumerate(chunks):
    #     print("="*100)
    #     print(f"{i+1}. {doc.metadata['source']}")
    #     print(f"{doc.page_content}\n")
    #     print("="*100)
        
    # Step 6 : Feed documents to model
    model_response = get_llm_response(documents=chunks, query=query)
    
    print(model_response)

if __name__ == "__main__":
    main()