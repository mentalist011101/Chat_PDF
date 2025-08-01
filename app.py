# This is your app.py file

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css, bot_template,user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import torch

from dotenv import load_dotenv
import os
load_dotenv()


def download_embeddings():
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(
      model_name= model_name,
      model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
      )
  return embeddings

def get_pdf_text(pdf_docs):
  "Recuperer le contenu de tous les PDfs et combiner en une seule variable"
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()

  return text

def get_text_chuncks(text):
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200, # Corrected typo here
      length_function = len
  )
  chuncks = text_splitter.split_text(text) # Corrected method name here
  return chuncks

def get_vectorstore(text_chuncks):
  embeddings = HuggingFaceInstructEmbeddings(
      model_name="hkunlp/instructor-xl",
      model_kwargs={'use_auth_token': HUGGINGFACEHUB_API_TOKEN} # Pass token here
  )
  vectorstore = FAISS.from_texts(
      text = text_chuncks,
      embeddings = embeddings
  )
  return vectorstore

def get_conversation_chain(vector_store):
  #llm = ChatOpenAI()
  llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs = {"temperature":0.5, "max_length":512})
  memory = ConversationBufferMemory(memory_key='chat-history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vector_store.as_retriever(),
      memory=memory
  )
  return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
  #CONFIGURATION DE LA PAGE
  st.set_page_config(page_title="Chat avec tes PDFs", page_icon=":books:", layout="wide")

  st.write(css, unsafe_allow_html=True)

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None


  if "conversation" not in st.session_state:
    st.session_state.conversation = None



  st.header("Chat avec plusieurs PDFs :books:")
  user_question = st.text_input("Pose des questions sur tes documents:")

  if user_question:
    handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Tes documents")
    pdf_docs = st.file_uploader("Charger vos PDFs ici et cliquez sur 'Charger'", accept_multiple_files=True)

    if st.button("Charger"):
      with st.spinner("Recherche en cours..."):
        # Recuperation du contenu
        raw_text = get_pdf_text(pdf_docs)

        # Creer les chuncks
        text_chuncks = get_text_chuncks(raw_text)

        # Crerr les vectors store
        vectorstore = get_vectorstore(text_chuncks)

        # conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == "__main__":
    main()
