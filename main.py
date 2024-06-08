## Import Libraries and Setup
import numpy as np 
import os
import streamlit as st 
import tempfile 
import google.generativeai as genai 
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain.vectorstores.faiss import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate 
from PIL import Image  
from PyPDF2 import PdfReader 
from streamlit_option_menu import option_menu 


## Load the Environment Variables
load_dotenv() 
os.getenv("GOOGLE_API_KEY") 


## Configure Model Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
model = genai.GenerativeModel('gemini-1.5-pro-latest') 


## Streamlit Configuration
st.set_page_config(
    page_title= "BUMA Gemini Assistant",
    page_icon=":🚜:",
    layout="centered"
)

## Custom CSS
with open("design.css") as source_des: 
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

## Sidebar Menu
with st.sidebar: 
    selected = option_menu(menu_title="BUMA Gemini Assistant",
                           options=["ChatBot",
                                    "File Assistant",
                                    "CSV Assistant"
                                    ],
                            menu_icon='robot', icons=['chat-dots-fill', 'file-earmark-pdf-fill', 'pc-display-horizontal'],
                            default_index=0
                           )

## Translates the user role ("model" or "user") to the appropriate value for Streamlit's chat message display
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

## Extracts the text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text += "".join(page.extract_text() for page in pdf_reader.pages)
    return text

## Splits the text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

## Creates a FAISS vector store from the text chunks and saves it locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

## Configures the ChatGoogleGenerativeAI model and creates a question-answering chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

##  Loads the FAISS vector store, performs a similarity search on the user's question
def user_input(user_question, model):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']


## Main Chatbot Function
def main_chatbot():
    st.image('img/BUMA.jpg')
    img = st.file_uploader("Choose an image",type=["jpg","jpeg","png"])

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)
    if st.button("Clear Chat"):
        st.session_state.chat_session = model.start_chat(history=[])
        st.success("Chat history cleared")

    if img is not None:
        image = Image.open(img)
        if st.button("Submit"):
            st.write("Thanks for uploading image")
            st.image(image=image,width=200)

        img_question = st.chat_input("ask anything")
        
        if img_question:
            st.chat_message("user").markdown(img_question)
            response = model.generate_content([img_question,image],stream=True)
            response.resolve()
            with st.chat_message("assistant"):
                st.markdown(response.text)

    else:
        user_prompt = st.chat_input("Ask me anything...")
        if user_prompt: 
            st.chat_message("user").markdown(user_prompt)
            gemini_response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text) 

## Main File Uploader Function
def main_file_uploader(): 
    st.image('img/BUMA.jpg')
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
    if st.button("Process PDF") and pdf_docs:
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF Processing Complete")

    user_question = st.chat_input("Ask any question from the PDF Files")
    if user_question:
        response = user_input(user_question, model)
        st.chat_message("user").markdown(user_question)
        with st.chat_message("assistant"):
            st.markdown(response)

## CSV Assistant            
if selected == "ChatBot": 
    main_chatbot()
elif selected == "File Assistant":
    main_file_uploader()
# elif selected == "CSV Assistant":
#     main_csv()