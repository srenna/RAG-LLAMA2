# Retrieval-Augmented Generation with LLAMA2 

This Streamlit app called "Point of View Hub" allows users to either scrape text from a URL or upload a PDF. It then provides a chatbot interface where users can ask questions related to the scraped content, and the chatbot uses a language model to generate responses based on the provided text.
The RAG (Retrieval-Augmented Generation) model blends retrieval-based and generation-based approaches in natural language processing. It employs a retrieval component to identify relevant documents from a corpus based on the input query, then utilizes a generation component to produce the final answer. This integration allows RAG models to leverage pre-existing knowledge effectively, resulting in accurate and informative responses across a wide range of queries. By combining the strengths of both retrieval and generation techniques, RAG models offer improved performance and flexibility for tasks such as question answering.

The app integrates several technical components:

1. **User Interface (UI)**:
   - It's built using Streamlit, a Python library for creating interactive web applications.
   - The UI includes options for scraping text from a URL or uploading a PDF file.
   - A chatbot interface allows users to interact with the system by asking questions.

2. **Data Scraping**:
   - For scraping text from a URL, the app uses the `requests` library to fetch the HTML content, and `BeautifulSoup` for parsing the content and extracting text.
   - For PDF uploads, it uses `PyPDF2` to extract text from the uploaded PDF.

3. **Language Model**:
   - The app employs a language model for generating responses to user queries. Specifically, it uses a model called "TheBloke/Llama-2-13B-chat-GPTQ" (LLAMA2). This model is based on GPT (Generative Pre-trained Transformer) architecture, specialized for conversational question-answering tasks.
   
4. **Pipeline and Embeddings**:
   - It uses `langchain` library to set up the pipeline for processing text and generating responses.
   - Embeddings from Hugging Face models are used for text representations.
   - A retrieval-based question-answering (QA) system is implemented for responding to user queries, possibly involving multiple documents.

Overall, the app provides a user-friendly interface for interacting with a sophisticated language model, capable of answering queries based on the scraped content, especially tailored for Deloitte-related topics.

## How to Run in Google Collab 

The notebook "streamlit_via_collab" allows for a Streamlit app running on port 8501 in the background, then use localtunnel to expose it to the internet. Upload the app.py file and Deloitte logo and run the command.

```streamlit run app.py & npx localtunnel --port 8501```

## Teammates
| Name             | Email                  | Socials                                      |
|------------------|------------------------|-------------------------------------------------------|
| Zabih Buda | zzabih@schulich.yorku.ca | [LinkedIn](https://www.linkedin.com/in/zabih-buda-3b9b6076/)      |
| Sabrina Renna    | srenna@schulich.yorku.ca | [LinkedIn](https://www.linkedin.com/in/sabrinarenna/)               |
| Vivek Periera  | vivekp@schulich.yorku.ca | [LinkedIn](https://www.linkedin.com/in/vivek-pereira/)        |
