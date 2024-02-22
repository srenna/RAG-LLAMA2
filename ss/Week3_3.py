# Imports for UI
import streamlit as st
import pdfplumber
import tempfile
import os
import requests
from bs4 import BeautifulSoup
import csv
from streamlit_option_menu import option_menu

# Imports for LLM
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#https://stackoverflow.com/questions/76431655/langchain-pypdfloader
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

DEFAULT_SYSTEM_PROMPT = """
You are a consultant from Deloitte. Answer the questions with details related to the contents found in the text files.""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#Implementation of a horizontal navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "POV Chatbot"],
    icons=["house", "robot"],
    orientation="horizontal",
    styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#86BC24", "font-size": "15px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#0F0B0B"},
            }
)

# Define a function to scrape text from a URL
def scrape_text_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the div with ID "article-title"
        article_title = soup.find('div', {'id': 'article-title'})
        article_title_text = article_title.get_text() if article_title else ''

        # Extract text from the div with class "content-components"
        content_components = soup.find('div', {'class': 'content-components'})
        content_components_text = content_components.get_text() if content_components else ''

        # combining the title and content
        article_text = article_title_text + content_components_text

        # split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        split_article_text = text_splitter.split_text(article_text)

        return split_article_text
    
    else:
        # Print an error message if the request was unsuccessful
        st.error(f"Failed to retrieve content. Status code: {response.status_code}")
        return None

#Code for the home page
if selected == "Home":
    st.title(f"{selected}")

#All of the code to use the chatbot is in here 
if selected == "POV Chatbot":
    st.title(f"{selected}")

    
    # Button to trigger LLM code
    if st.button("Run LLM"):
        # LLM Code
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
        )
        model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        model_basename = "model"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            revision="gptq-4bit-128g-actorder_True",
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False,
            device=DEVICE,
            quantize_config=None,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=10,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 10})
        st.success("LLM initialized successfully!")

    # Create a text input field for the user to enter the URL
    url_to_scrape = st.text_input("Enter URL to scrape:", "")

    # Create a button to trigger scraping
    if st.button("Scrape URL"):
        if url_to_scrape:
            result = scrape_text_from_url(url_to_scrape)
            if result:
                st.success("Text scraped successfully from URL!")
                st.write(result)
            else:
                st.error("Scraping failed. Please check the URL.")
        else:
            st.warning("Please enter a URL to scrape.")

    # Create a button to trigger PDF text scraping
    uploaded_file = st.file_uploader("Upload your PDF")
    if uploaded_file is not None:
        if st.button("Scrape PDF"):
            docs = []
            reader = PdfReader(uploaded_file)
            i = 1
            for page in reader.pages:
                docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
                i += 1
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
            result = text_splitter.split_documents(docs)
            st.write(result)
            db = Chroma.from_documents(result, embeddings, persist_directory="db")
                
            qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            )



    # This code is an integration of what we had originally and the code that Sabrina had to show coversation history
    # Define your conversational_chat function
    def conversational_chat(query, chat_history):
        # Replace this with your actual chatbot logic
        response = "Response to user"
        chat_history.append(f'User: {query}')
        chat_history.append(response)
        return response

    # Initialize session_state if not already present
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Please enter your query in the textbox below"]

    if 'past' not in st.session_state:
        st.session_state['past'] = [""]

    # Create Streamlit containers for chat history and user input
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Enter your query here")
            submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # Update the chat history
        conversational_chat(user_input, st.session_state['history'])
        st.session_state['past'].append(f'You: {user_input}')
        result = qa_chain(user_input)
        st.session_state['generated'].append(result)

    with response_container:
        # Display the chat history
        for i in range(len(st.session_state['generated'])):
            st.text(st.session_state['past'][i])
            st.text(st.session_state['generated'][i])
