from dotenv import load_dotenv
import os
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수를 사용하여 API 키를 불러옵니다.
openai_api_key = os.getenv('OPENAI_API_KEY')

# 모델 초기화
llm = ChatOpenAI(openai_api_key=openai_api_key)

# langchain 모델 기본 사용
# output = llm.invoke("2024년 청년 지원 정책에 대하여 알려줘")
# print(output)


# Template 기반 사용법
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야"),
    ("user", "{input}")
])

chain = prompt | llm    # prompt와 llm을 |(pipe)로 연결하는 python 문법 - prompt를 llm 함수의 인자로 넘겨주는 작업

print(chain)

print(chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘"}))