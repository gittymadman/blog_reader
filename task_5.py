'''
This will act as a home page to my streamlit application in the pages folder, here you can start chatting with the AI and when you click New Chat in the chatting page, your previous chat will be stored in the Latest Chat. To start chatting again, click on Let's go
'''

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
# import sqlite3
from langchain_core.messages import AIMessage,HumanMessage
# conn = sqlite3.connect('example.db') 
st.title("Main Page")
st.header("Welcome VK, Shall we start !!")
st.header("Press the button to continue")
st.sidebar.header("Previous Chats")

if "l" not in st.session_state:
    st.session_state.l = []


with st.sidebar:
    w = st.button(label="Latest Chat")
    st.session_state.l = st.session_state.chat_history.copy()
    if w:
        for message in st.session_state.chat_history:
            if isinstance(message,AIMessage): # If the message is an instance of the AIMessage object, it gets printed
                with st.chat_message("AI"):
                    st.write(message.content)     
                    
            elif isinstance(message,HumanMessage): # If the message is an instance of the HumanMessage object, it gets printed
                with st.chat_message("Human"):
                    st.write(message.content)
    
    
x = st.button(label="Let's Go")
if x:
    
    st.session_state.chat_history = [
    AIMessage(content="Hellow how can I help you")]
    st.session_state.history = ""
    switch_page("home_page")


    

    
    
    