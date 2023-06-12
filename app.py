import streamlit as st
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
import os

# Create the sidebar
with st.sidebar:
    st.title("PDF_ChatGPT: An LLM chat app")
    st.markdown('''
    ## About
    PDF_ChatGPT is an app written in python built with the following:
    - [streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI LLM model gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5) 
    - [Facebook AI Similarity Search (FAISS)](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html)
    ''')
    add_vertical_space(5)
    st.write('Authored by c0atl with help from youtube')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def main():
    load_dotenv()
    st.header("Chat with any PDF file using ChatGPT and Langchain 🧠")
    folder_path = "./pdfs"
    # accept a pdf file the user uploads
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)

    if selected_filename is not None:
        # load and split the document by page
        pdf_path = os.path.join(folder_path, selected_filename)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
 
        # create embeddings from pages
        vector_store = FAISS.from_documents(pages, OpenAIEmbeddings())
        store_name = selected_filename[:-4]
        
        # only create an embedding vector if one doesn't exist already
        if os.path.exists(f"embeddings/{store_name}.pkl"):
            with open(f"embeddings/{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write(f"Using existing embedding '{store_name}.pkl' found on disk. Remove it to create a new embedding.")
        else:
            with open(f"embeddings/{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write(f"Created new embedding 'embeddings/{store_name}.pkl'")

        # create prompt for user questions
        query = st.text_input(f"What do you want to know about {selected_filename}?")

        if query:
            # return the k most relevant chunks of the pdf text
            docs = vector_store.similarity_search(query=query, k=4) # k default is 4
            llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
            chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")
            # check cost of operation
            with get_openai_callback() as cb:
                response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                print(cb)

            source_pages = ''
            for i in range(len(docs)):
                print(docs[i].metadata['page'])
                source_pages += str(docs[i].metadata['page'])
                if i < len(docs):
                    source_pages += ', '

            st.write(f"Response Data taken from pages: {source_pages}:\n {response['output_text']}")

            
if __name__=='__main__':
    main()