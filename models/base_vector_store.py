from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os


def get_vector_store(model_name: str, name: str, pdf: str):
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)

    if os.path.exists("./models/chroma_db"):
        store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name=name
        )
        return store

    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()
    store = Chroma.from_documents(
        pages,
        embeddings,
        collection_name=name,
        persist_directory="./models/chroma_db"
    )
    return store

if __name__ == "__main__":
    model_name = "thenlper/gte-base"
    store = get_vector_store(
        model_name=model_name,
        name="basic-laws",
        pdf='./data/basic-laws-book-2016.pdf')
    print(store.similarity_search("law of the united states", k=5))
