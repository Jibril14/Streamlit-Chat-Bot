
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain import PromptTemplate
import os
import streamlit as st


import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from dotenv import load_dotenv
load_dotenv(".env")


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, you should say that 'I've searched my database, but I couldn't locate the exact information you're looking for. However, some of the documents did mention part of the keywords as listed. Would you like me to broaden the search and provide related information that might be helpful?', don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT_ERROR = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
import os
OPENAI_API_KEY =  os.environ["OPENAI_API_KEY"] = "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"

memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), max_token_limit=150, memory_key='chat_history', return_messages=True, output_key='answer')


st.title("👋 ChatBot for Learning About American Laws")

	
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = memory



# connect to a Qdrant Cluster
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embeddings = OpenAIEmbeddings()

vector_store = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)
# docs = vector_store.similarity_search("Law of city of Maricopa", k=1)
# print("Type Doc",type(docs))



qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                           vector_store.as_retriever(), memory=st.session_state.buffer_memory,
                                           verbose=True,
                                           return_source_documents=True,
					   combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR})

result = qa({"question": "is America the largest country"})
print("Result", result)
# query = "Is United state of America largest"
# response = qa.run(query)
# print("Res: ",response) # 

st.write(result)
