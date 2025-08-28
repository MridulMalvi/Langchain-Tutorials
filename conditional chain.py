from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import *
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableBranch ,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
import os

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class FeedbackModel(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=FeedbackModel)
 
prompt1 = PromptTemplate(
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
    template="classify sentiment of feedback into positive or negative: {feedback}\n{format_instructions}"
)

prompt2 = PromptTemplate(
    input_variables=["feedback"],
    template="Write a appropriate response to this positive feedback: {feedback}"
)

prompt3 = PromptTemplate(
    input_variables=["feedback"],
    template="Write a appropriate response to this negative feedback: {feedback}"
)

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
   RunnableLambda(lambda x: "No response found for sentiment ")
)

chain=classifier_chain | branch_chain

print(chain.invoke({"feedback": "I love the new features of this product!"}))