import re
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain import PromptTemplate
import os

import time

import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from dotenv import load_dotenv
load_dotenv(".env")

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't
 know the answer, you should say that 'I've searched my database, but I couldn't locate the
 exact information you're looking for. However, some of the documents did mention 
 part of the keywords you entered. May be you want to be more specific in your search.' 
 then try to make up an answer base on the context only.
 
Context: {context}
Question: {question}
Helpful Answer:"""
QA_PROMPT_ERROR = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


memory = ConversationSummaryBufferMemory(
    llm=OpenAI(
        temperature=0),
    max_token_limit=150,
    memory_key='chat_history',
    return_messages=True,
    output_key='answer')


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


def get_urls(doc):
    url_regex = '(http[s]?://?[A-Za-z0-9–_\\.\\-]+\\.[A-Za-z]+/?[A-Za-z0-9$\\–_\\-\\/\\.\\?]*)[\\.)\"]*'
    url = re.findall(url_regex, doc)
    return url


def print_answer_citations_sources(result):
    links = []
    output_answer = ""
    output_answer += result['answer']
    for doc in result['source_documents']:
        link = get_urls(doc.page_content)
        links.extend(link)
    link = "\n".join(links)
    # print("Link OUT",links)
    # print("Lin OUT",link)
    if links != []:
        output_answer += "\n" + "See also: " + link

    print("OUT", output_answer)

    return output_answer


def extract_page_content_and_title(result):
    
    extracted_string = ""

    
    for doc in result['source_documents']:
       
        page_content = doc.page_content[:200] + "..."
       
        title = doc.page_content[0:30]
        if page_content and title:
            extracted_string += f"----------------------\n\nDocument Title: {title}\n\n\n Excerpt: {page_content}\n\n"

    return extracted_string


qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vector_store.as_retriever(),
    memory=st.session_state.buffer_memory,
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR})


response_container = st.container()

textcontainer = st.container()

details = ''


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        res = qa({"question": query})
        response = print_answer_citations_sources(res)
        details = extract_page_content_and_title(res)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)


# count = 0
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(
                    st.session_state["requests"][i],
                    is_user=True,
                    key=str(i) + '_user')

with st.sidebar:
    if details:
        with st.spinner("Processing..."):
            time.sleep(1)
            st.write("Similar Documents")
        st.write(details)
