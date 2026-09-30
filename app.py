# Created by Robson Elias Goncalves
import streamlit as st
import os
import time

# userprompt
from langchain_core.prompts import PromptTemplate

# vectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# llm
from langchain_community.llms import Ollama

# pdf loader
from langchain_community.document_loaders import PyPDFLoader

# pdf processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

# retrieval (imports removed - using simple function instead)

if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')


if 'template' not in st.session_state:
    st.session_state.template = """You area a knowledge chatbot, here to help with questions of the user. The tone should be a professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot: """

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ""

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='vectorDB',
                                          embedding_function=HuggingFaceEmbeddings()
                                          )
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url='http://localhost:11434',
                                  model="mistral",
                                  verbose=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Chatbot - to talk do PDFs")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    st.text("File uploaded successfully")

    if 'processed_file' not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.status("Saving file..."):
            if not os.path.exists('pdfFiles/' + uploaded_file.name):
                bytes_data = uploaded_file.read()
                f = open('pdfFiles/' + uploaded_file.name, 'wb')
                f.write(bytes_data)
                f.close()

            loader = PyPDFLoader('pdfFiles/' + uploaded_file.name)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )

            all_splits = text_splitter.split_documents(data)

            if not all_splits:
                st.error(
                    "No text was extracted from the PDF. The file may be scanned or image-based (without selectable text).")
                st.stop()

            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=HuggingFaceEmbeddings()
            )

            st.session_state.vectorstore.persist()
            st.session_state.processed_file = uploaded_file.name

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
        def qa_function(question, history):
            docs = st.session_state.retriever.invoke(question)
            context = "\n".join(
                [doc.page_content for doc in docs]) if docs else "No relevant documents found."

            full_prompt = st.session_state.prompt.format(
                context=context,
                history=history,
                question=question
            )

            response = st.session_state.llm.invoke(full_prompt)
            return response

        st.session_state.qa_chain = qa_function

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                history_str = "\n".join(
                    [f"{msg['role']}: {msg['message']}" for msg in st.session_state.chat_history[:-1]])
                response = st.session_state.qa_chain(user_input, history_str)

            message_placeholder = st.empty()
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)

                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a PDF file to start the chatbot")
