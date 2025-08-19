from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

                )
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Give detailed report on {question}",
    input_variables=["question"])
    
template2 = PromptTemplate(
    template="Give 5 line summary of {question}?",
    input_variables=["question"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"question": "Deforestation"})

print(result)