from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import EMBEDDING_MODEL


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def build_retriever(pdf_path: str):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )