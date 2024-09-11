from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests
from langchain_core.messages import AIMessage,HumanMessage
# from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.document_transformers import LongContextReorder # To avoid Lost in Middle Problem
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity # Used here for checking whether next question is related to previous output of LLM
from sentence_transformers import SentenceTransformer , util
from streamlit_extras.switch_page_button import switch_page



from pypdf import PdfReader

def data_initial_load():
    #Loading Data
    data=''
    pdf = PdfReader("blog.pdf")
    for pages in pdf.pages:
        data += pages.extract_text()

    # loader = WebBaseLoader("https://en.wikipedia.org/wiki/Realme")
    # data = loader.load() # Getting text in documetn form
    # # st.write(data)
    return data

def embedding(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 30)
    docs = splitter.split_text(data)
    # st.write ("DOCS : ",docs)

    #st.write(docs)
    # Creating Embedding model
    hf = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    #Creating Vector Database
    db = Chroma.from_texts(docs, hf)
    # st.write(db)
    st.write("Embeddings completed")
    return db


def chat_prompt_tempalte(input,history=None):
    if history:
        template =  '''Here is the previous content
            content = {history}
            User question: {input}\n\n
            Restrict yourself to the content only
            Do not mix the content and the User question for giving answer.
            Just give output as no if you can not answer the question from content or else yes. Be Strict on that'''
        return template.format(input=input,history=history)
    else:
        template = '''User question: {input}\n\n
            restrict yourself to answers realted to input from database
            Provide a accurate response based on the input context only.If the contenxt or question is not fully relevant, do not answer at all.'''
        return template.format(input = input)


def answer(db,input_new):
    docs = db.similarity_search(query=input_new, k=5)  # Getting more documents initially to rerank/reorder
    #st.write(docs)
    # Reordering
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    relevant_docs = [doc.page_content for doc in reordered_docs]
            
            
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512) # Huggingface model
    unique_docs = []
    scores = []
    #x = 0
    for doc in relevant_docs:
        unique_docs.append(doc)
        scores.append(model.predict([input_new,doc]))
        #x += 1
                
        scored_docs = zip(scores, unique_docs)
        sorted_docs = sorted(scored_docs, reverse=True)
        reranked_docs = [Document(page_content=doc) for _, doc in sorted_docs][0:2]
        #st.write(reranked_docs)
# http://192.168.1.4:11434/
    llm = Ollama(
        base_url='http://localhost:11434/',
        model="llama3",
        temperature = 0)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # print(reranked_docs)
    response = chain.run(input_documents=reranked_docs, question=input_new)
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hellow how can I help you")
    ]

if "history" not in st.session_state:
    st.session_state.history="" # To store previous user input and output provided by llm to ask question based on previous output

## Page Configurations ##
st.header("Let's chat about How to write a blog :full_moon_with_face:")

st.sidebar.header("Click below for New Chat")
x_new = st.sidebar.button(label="New Chat")
if x_new:
    switch_page("task_5")


input = st.chat_input("Type your question here ...") # User Input

data = data_initial_load()
if input is not None and input !="":
    if st.session_state.history =="":
        st.write("Enter 1234567")
        db = embedding(data)
        input_new = chat_prompt_tempalte(input)
        response = answer(db,input_new)
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=input))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.history = response # Updating with lastest output
    else:
        # st.write("HISTORY",st.session_state.history)
        db_new = embedding(st.session_state.history)
        st.write("DB_NEW created")
        input_new = chat_prompt_tempalte(input,st.session_state.history)
        response = answer(db_new,input_new)
        st.write("Response is :",response)
        st.write("asdfghjklxcvbnmert")
        if "No" or "no" in response:
            st.write("ENtered here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            db = embedding(data)
            input_new = chat_prompt_tempalte(input)
            response = answer(db,input_new)
        st.write("did not enter ::((((((((((((((((((((((()))))))))))))))))))))))")
        st.session_state.chat_history.append(HumanMessage(content=input))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.write("Hisotry is :",st.session_state.history)
        st.session_state.history = response # Updating with lastest output
        
for message in st.session_state.chat_history:
        
        if isinstance(message,AIMessage): # If the message is an instance of the AIMessage object, it gets printed
            with st.chat_message("AI"):
                st.write(message.content)     
        
        elif isinstance(message,HumanMessage): # If the message is an instance of the HumanMessage object, it gets printed
            with st.chat_message("Human"):
                st.write(message.content) 