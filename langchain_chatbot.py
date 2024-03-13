from dotenv import load_dotenv
import os
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수를 사용하여 API 키를 불러옵니다.
openai_api_key = os.getenv('OPENAI_API_KEY')

# main.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st

API_KEY = os.getenv('OPENAI_API_KEY') 
MODEL = "gpt-4-0125-preview"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

content="""# 프롬프트 엔지니어링

# 필요성

프롬프트 엔지니어링이란 인공지능(AI) 언어 모델에게 특정한 출력을 유도하기 위해 입력 텍스트(프롬프트)를 세심하게 설계하는 과정을 의미합니다. 이는 AI가 다양한 작업을 수행하도록 하기 위해 필수적인 과정으로, 최근 OpenAI의 GPT와 같은 고급 언어 모델의 등장으로 그 중요성이 더욱 부각되고 있습니다. 프롬프트 엔지니어링의 필요성은 다음과 같은 몇 가지 이유에서 기인합니다:"""

st.header("백엔드 스쿨/파이썬 2회차(9기)")
st.info("프롬프트 엔지니어링에 대한 내용을 알아볼 수 있는 Q&A 로봇입니다.")
st.error("프롬프트 엔지니어링에 대한 내용이 적용되어 있습니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 백엔드 스쿨 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():   
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))