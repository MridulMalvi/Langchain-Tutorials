from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence 

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

                )
model = ChatHuggingFace(llm=llm)
prompt= PromptTemplate(
    template ="Write a joke about {topic}",
    input_variables=["topic"])

parser = StrOutputParser()

chain = RunnableSequence (prompt , model , parser)
print(chain.invoke({'topic':'AI'}))