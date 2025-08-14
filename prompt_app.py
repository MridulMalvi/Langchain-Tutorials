
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.header("Research tool")
user_input = st.text_input("Enter prompt")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

if st.button("Summerize"):
    result=model.invoke(user_input)
    st.write(result.content)

# llm = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    # task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)
# 
# result = model.invoke("Explain deforestation?")
# print(result.content)
# 