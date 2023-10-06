import os
import time
import re
import random
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain import PromptTemplate
import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from dotenv import load_dotenv
load_dotenv(".env")


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't
 know the answer, or similar answer is not in the context, you should say that 'I've searched my database,
 but I couldn't locate the exact information you're looking for. May be you want to be more specific
 in your search. Or checkout similar documents'.
 Answer user greetings and ask them what they i'd like to learn about. You are a bot that teaches users
 about american law codes

Context: {context}
Question: {question}
Helpful Answer:"""
QA_PROMPT_ERROR = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Shuffle logo
def logo(logo: str = None):
    logos = [

        "https://res.cloudinary.com/webmonc/image/upload/v1696515089/3558860_r0hs4y.png"
    ]
    logo = random.choice(logos)
    print("LOGO", logo)
    return logo


memory = ConversationSummaryBufferMemory(
    llm=OpenAI(
        temperature=0),
    max_token_limit=150,
    memory_key='chat_history',
    return_messages=True,
    output_key='answer')


# Streamlit Component

st.set_page_config(
    page_title="USA Law Codes",
    # page_icon=":robot:"
    page_icon=":us:"
)

st.header("📋 ChatBot for Learning About USA Laws")
# st.title("👋 📝 ChatBot for Learning About American Laws")
user_city = st.selectbox("Select a City", ("Maricopa", "LAH"))
# user_chat = st.text_input("You: ", key=input)
# submit = st.button("Browse Law Code")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["I'm here to assist you!"]

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


def connect_db(db=None):
    # db = os.getenv("QDRANT_COLLECTION_NAME")
    db = user_city
    if user_city == "LAH":
        db = "collection_two"  # I.e set a collection/DB name
    elif db == "Maricopa":
        db = "test3"

    vector_store = Qdrant(
        client=client,
        collection_name=db,
        embeddings=embeddings
    )
    return vector_store


def get_urls(doc):
    url_regex = '(http[s]?://?[A-Za-z0-9–_\\.\\-]+\\.[A-Za-z]+/?[A-Za-z0-9$\\–_\\-\\/\\.\\?]*)[\\.)\"]*'
    url = re.findall(url_regex, doc)
    return url


def print_answer_metadata(result):
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


def print_page_content(result):

    extracted_string = ""

    for doc in result['source_documents']:

        page_content = doc.page_content[:200] + "..."

        title = doc.page_content[0:35] + "..."
        if page_content and title:
            extracted_string += f"<hr><h4>Document Title:</h4> {title}\n\n\n <h4>Excerpt:</h4>\
                {page_content}\n\n"

    return extracted_string


qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    connect_db().as_retriever(),
    memory=st.session_state.buffer_memory,
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR})


response_container = st.container()
textcontainer = st.container()

details = ''


with textcontainer:
    query = st.text_input("You: ", key="input", placeholder="start chat")
    submit = st.button("send")
    if submit:
        res = qa({"question": query})
        response = print_answer_metadata(res)
        details = print_page_content(res)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)


with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(
                st.session_state['responses'][i],
                key=str(i),
                avatar_style="no-avatar",
                logo=logo(),
                allow_html=True)
            if i < len(st.session_state['requests']):
                message(
                    st.session_state["requests"][i],
                    is_user=True,
                    key=str(i) + '_user',
                    allow_html=True
                )


with st.sidebar:
    st.image("https://res.cloudinary.com/webmonc/image/upload/v1696603202/Bot%20Streamlit/law_justice1_yqaqvd.jpg")
    if details:
        with st.spinner("Processing..."):
            time.sleep(1)
            st.markdown('__Similar Documents__')
        # st.write(details)
        # st.markdown('''<hr>''', unsafe_allow_html=True)
        st.markdown(f'''<small>{details}</small>''', unsafe_allow_html=True)
