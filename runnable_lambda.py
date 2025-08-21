from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence ,RunnableParallel ,RunnableLambda ,RunnablePassthrough

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

                )
model = ChatHuggingFace(llm=llm)

def count(text):
    return "function called"

parser= StrOutputParser()

prompt1= PromptTemplate(
    template ="Write a joke about {topic}",
    input_variables=["topic"])

joke_chain =RunnableSequence(prompt1 ,model ,parser)


parallel_chain =RunnableParallel( {"joke" : RunnablePassthrough(),
                            "count": RunnableLambda(count)
 })
final_chain=RunnableSequence(joke_chain,parallel_chain)
result =final_chain.invoke({"topic":"Humans"})
final_result ="""{}\n function - {}""".format(result["joke"],result["count"])
print (final_result)
