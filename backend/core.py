import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

from prompt import qa_prompt
from utils.constants import Constants

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


def execute_chat(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeLangChain.from_existing_index(
        embedding=embeddings,
        index_name=Constants.INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt.generate_chat_prompt()},
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(execute_chat(query="Query input", chat_history=[]))
