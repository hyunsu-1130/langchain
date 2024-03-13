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


# 바로 Docs 내용을 반영도 가능합니다.
from langchain_core.documents import Document

print(document_chain.invoke({
    "input": "국민취업지원제도가 뭐야",
    "context": [Document(page_content="""국민취업지원제도란?

취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
[출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24""")]
}))