import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from typing import List

from utils.constants import Constants

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


def process_documents(path, chunk_size, chunk_overlap, index, embedding_model):
    documents = load_docs(path)
    split_documents = split_docs(documents, chunk_size, chunk_overlap)
    embedding_docs(embedding_model, index, split_documents)


def load_docs(path: str) -> List[Document]:
    loader = PyPDFLoader(path)

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw documents")
    return raw_documents


def split_docs(documents, chunk_size, chunk_overlap) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into chunks.")
    return documents


def embedding_docs(model, index, documents):
    embeddings = OpenAIEmbeddings(model=model)
    print(f"Adding {len(documents)} documents to Pinecone")
    PineconeLangChain.from_documents(documents, embeddings, index_name=index)
    print("Vectorstore loading completed.")


if __name__ == "__main__":
    process_documents(
        Constants.DOCUMENT_PATH,
        600,
        50,
        Constants.INDEX_NAME,
        Constants.EMBEDDING_MODEL,
    )
