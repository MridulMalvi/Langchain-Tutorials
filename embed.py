from langchain_huggingface import  HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedd = HuggingFaceEmbeddings(   model_name='sentence-transformers/all-MiniLM-L6-v2')


text="Hello, how are you?"
vector= embedd.embed_query(text)

print(str(vector))

