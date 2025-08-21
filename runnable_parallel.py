from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence ,RunnableParallel

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

                )
model = ChatHuggingFace(llm=llm)
prompt1= PromptTemplate(
    template ="Write a joke about {topic}",
    input_variables=["topic"])

prompt2= PromptTemplate(
    template ="Describe only in 2 lines: {topic}",
    input_variables=["topic"])

parser = StrOutputParser()

parallel_chain =RunnableParallel( {"joke" :RunnableSequence (prompt1 , model , parser),
                              "Description":  RunnableSequence (prompt2 , model , parser)   })

print(parallel_chain.invoke({'topic':'AI'}))