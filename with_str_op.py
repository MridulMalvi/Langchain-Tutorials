

from typing import TypedDict, Annotated
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os; 

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

class review(TypedDict):
    summary: Annotated[str , "The brief summary of the review"]
    sentiment:Annotated[str ,"The sentiment of the review, either positive , negative or neutral"]
structured_model =model.with_structured_output(review)
    
result = structured_model.invoke("""A "4 line text" can refer to different things depending on the context. It could mean a block of text that is intentionally limited to four lines, or it could refer to the specific formatting of a text document with four lines per page. Additionally, it can refer to a text input field that allows up to four lines of text.""")
print(result)
