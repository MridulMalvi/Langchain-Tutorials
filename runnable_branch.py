from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence ,RunnableParallel ,RunnableLambda ,RunnablePassthrough ,RunnableBranch

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                )

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1= PromptTemplate(input_variables=["Topic"],
            template="Describt the tpic {Topic}"            )

prompt2= PromptTemplate(input_variables=["desc_Topic"],
                        template="Summerize the text in 200 words {desc_Topic}")

desc = RunnableSequence(prompt1 , model ,parser)

summeri=RunnableBranch(
(lambda x:len(x.split())>200 ,RunnableSequence(prompt2,model ,parser))    ,#condition
RunnablePassthrough()    #default
)

Final_chain=RunnableSequence(desc ,summeri)
print(Final_chain.invoke({"Topic":"the wildlife"}))






