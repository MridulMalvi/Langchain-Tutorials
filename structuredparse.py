from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser ,ResponseSchema

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

                )
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 :"),
    ResponseSchema(name="fact_2", description="Fact 2 :")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template  = PromptTemplate(
    template='Give 2 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = template | model | parser
# prompt = template.invoke({'topic': "Artificial Intelligence"})
# result = model.invoke(prompt)
#Final_result =parser.parse(result.content)

result = chain.invoke({'topic': "Artificial Intelligence"})
print(result)