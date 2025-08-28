from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import *
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
import os

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["input"],
    template="Give name of: {input}"
)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"input": "First person to reach on moon"})
print(result)

chain.get_graph().print_ascii()

