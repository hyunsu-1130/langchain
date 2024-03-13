from dotenv import load_dotenv
import os
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import WebBaseLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain

# FAISS : 효율적인 유사성 검색 및 클러스터링의 대규모 데이터 셋을 위한 라이브러리 / 유사성 바탕 빠른 검색 속도 및 높은 정확도 제공
# Embedding : 사람이 이해하는 자연어나 이미지 등의 복잡한 데이터를 컴퓨터가 처리할 수 있는 숫자 형태의 벡터로 변환하는 기술

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수를 사용하여 API 키를 불러옵니다.
openai_api_key = os.getenv('OPENAI_API_KEY')

# 모델 초기화
llm = ChatOpenAI(openai_api_key=openai_api_key)

# 사이트 크롤링 및 로드
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")

docs = loader.load()

# 임베딩 초기화 ? # 크롤링한 정보를 chatGPT(AI)에 전달
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Template 기반 prompt 생성
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# 크롤링 사이트 내용 기반 언어처리 가능 !!!
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "국민취업지원제도가 뭐야"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...