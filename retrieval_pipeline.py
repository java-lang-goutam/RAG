from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history = []


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
        HumanMessage(content=llm_input),
    ]

    llm_response = model.invoke(messages)

    return llm_response.content


def ask_question(query):

    llm_model = get_llm_model()
    embedding_model = get_embedding_model()

    print(f"You asked : {query}")
    search_question = query

    if chat_history:
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {query}")]
        )

        response = llm_model.invoke(messages)
        search_question = response.content
        print(f"Searching for : {search_question}")

    persist_direcotry = "db/chroma_db"

    # Step 2 : Load vector store
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_direcotry,
        collection_configuration={"hnsw:space": "cosine"},
    )

    # Step 3: Configure retriever to load chunks
    vector_store_retriver = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )

    # Step 4 : Load chunks
    documents = vector_store_retriver.invoke(search_question)

    # Step 5 : Get LLM Response
    return get_llm_response(documents=documents, query=search_question)


def main():
    # Start ingestion pipeline
    print("Ask me questions! type 'quit' to exit")

    while True:
        print("=" * 100)
        question = input("\nYour question: ")
        if question.lower() == "quit":
            print("Good Bye!")
            break

        response = ask_question(question)
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        print(response)


if __name__ == "__main__":
    main()
