from langchain_huggingface import  HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embedd = HuggingFaceEmbeddings(   model_name='sentence-transformers/all-MiniLM-L6-v2')
doc=["Who is the only player to score 100 international centuries?",
"In which year did India win its first Cricket World Cup?",
"Who holds the record for the fastest century in ODI cricket?"
]
text='Who holds the record for the fastest century?'
doc_emb =embedd.embed_documents(doc)
text_emb = embedd.embed_query(text)
print( cosine_similarity([text_emb], doc_emb))
