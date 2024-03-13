from dotenv import load_dotenv
import os
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import WebBaseLoader

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수를 사용하여 API 키를 불러옵니다.
openai_api_key = os.getenv('OPENAI_API_KEY')

# 모델 초기화
llm = ChatOpenAI(openai_api_key=openai_api_key)

# 원하는 페이지 크롤링하기
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")

docs = loader.load()
print(docs)

