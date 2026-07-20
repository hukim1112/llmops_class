import os
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo


# 환경 변수 로드
load_dotenv(override=True)

# 경로 설정 (프로젝트 루트 기준)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
PDF_SOURCE_DIR = os.path.join(DATA_DIR, "bok_major_industry_reports")

# 전역 변수로 Retriever 관리 (Lazy Loading)
_retrievers = {
    "basic": None,
    "self_query": None,
    "multimodal": None
}

def _get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def _initialize_vectorstore(collection_name: str) -> Chroma:
    """
    지정된 Collection Name으로 Chroma VectorStore를 로드하거나 생성합니다.
    (실습 환경에서는 이미 생성되었다고 가정하거나, 없으면 로드 시도)
    """
    embedding_model = _get_embedding_model()
    
    # DB 디렉토리가 없으면 생성 (폴더만)
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    print(f"📂 [DataLoader] Loading VectorStore: {collection_name}")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    return vectorstore

# --- 1. Basic RAG Retriever ---
def get_basic_retriever():
    if _retrievers["basic"]:
        return _retrievers["basic"]

    vectorstore = _initialize_vectorstore("basic_rag")
    # Basic RAG는 일반적인 Similarity Search를 사용하는 Retriever
    _retrievers["basic"] = vectorstore.as_retriever()
    return _retrievers["basic"]

# --- 2. Self-Query Retriever (Shared Logic) ---
def _create_self_query_retriever(collection_name: str):
    vectorstore = _initialize_vectorstore(collection_name)
    
    metadata_field_info = [
        AttributeInfo(
            name="year",
            description="The year of the report (e.g., 2024). Must be an integer.",
            type="integer",
        ),
        AttributeInfo(
            name="quarter",
            description="The quarter of the report. One of 1, 2, 3, 4. Must be an integer.",
            type="integer",
        ),
    ]
    document_content_description = (
        "Bank of Korea Industry Reports. "
        "IMPORTANT: Filter values for 'year' and 'quarter' must always be "
        "integers (e.g., 2024, 1), NEVER strings (e.g., \"2024\", \"1\")."
    )
    
    # Self-Query를 위한 LLM (구조화된 쿼리 생성용)
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_config({"tags": ["exclude_from_stream"]})
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=False
    )
    return retriever

def get_self_query_retriever():
    if _retrievers["self_query"]:
        return _retrievers["self_query"]
        
    print("🛠 [DataLoader] Initializing Self-Query Retriever...")
    _retrievers["self_query"] = _create_self_query_retriever("self_query")
    return _retrievers["self_query"]

def get_multimodal_retriever():
    if _retrievers["multimodal"]:
        return _retrievers["multimodal"]

    print("🛠 [DataLoader] Initializing Multimodal Retriever (Self-Query enabled)...")
    _retrievers["multimodal"] = _create_self_query_retriever("multimodal")
    return _retrievers["multimodal"]


