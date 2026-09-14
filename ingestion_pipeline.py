from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os, sys


def print_documents(documents):
    for i, doc in enumerate(documents):
        print(
            f"{i+1}. {doc.metadata['source']}, len: {len(doc.page_content)} chars ({doc.page_content[:5]}...{doc.page_content[-5:]})"
        )


def check_db_exists(dir):
    if os.path.exists(dir):
        print("Vector store already exists")

        embedding_model = get_embedding_model()
        vector_store = Chroma(
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=dir,
        )

        print(f"Collection count : {vector_store._collection.count()}")
        sys.exit()


def load_documents(docs_path):
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No .txt files present in the directory!!")

    print("Loaded documents : ")
    print_documents(documents)

    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents=documents)

    print("Loaded chunks: ")
    print_documents(chunks)

    return chunks


def get_embedding_model():
    return OllamaEmbeddings(model="mxbai-embed-large")


def create_vector_store(chunks, persist_directory):
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        embedding=embedding_model,
        documents=chunks,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vector_store


def main():
    # Start ingestion pipeline
    print("Starting RAG Ingestion pipeline")
    persist_direcotry = "db/chroma_db"
    check_db_exists(persist_direcotry)

    # Step 1: Load directory / docs
    documents = load_documents(docs_path="docs")

    # Step 2 : Split the documents in chunks
    chunks = split_documents(documents)

    # Step 3 : Create embedding / vector store
    vector_store = create_vector_store(
        chunks=chunks, persist_directory=persist_direcotry
    )


if __name__ == "__main__":
    main()
