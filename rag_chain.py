from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = openai_api_key

# ▶ 문서 준비
raw_docs = [
    Document(page_content="Flask는 Python 웹 프레임워크입니다."),
    Document(page_content="ChromaDB는 벡터 검색을 위한 오픈소스 데이터베이스입니다."),
    Document(page_content="RAG는 검색 증강 생성 방식으로 LLM에 문서를 보강합니다."),
]

# ▶ 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_docs = text_splitter.split_documents(raw_docs)

# ▶ 임베딩 + 벡터스토어 저장
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embedding_fn)
vectordb.add_documents(split_docs)

# ▶ LLM 설정
llm = OpenAI(temperature=0)

# ▶ LangChain RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True
)

def get_rag_response(query: str) -> str:
    result = qa_chain({"query": query})
    return result["result"]