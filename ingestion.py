import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

INDEX_NAME = "INDEX_NAME"


def ingest_docs():
    loader = PyPDFLoader("DOCUMENT_PATH")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split {len(documents)} documents into chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Adding {len(documents)} documents to Pinecone")
    PineconeLangChain.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("Vectorstore loading completed.")


if __name__ == "__main__":
    ingest_docs()
