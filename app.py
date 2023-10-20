import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

#Rendering a Sidebar
with st.sidebar:
    st.title('PDF chat App')
    add_vertical_space(10)
    st.write('Made by [Girish Naik] (naikgirish711@gmail.com)')

#loading Env variables(Api key)
load_dotenv()

def main():
    st.header('Chat with PDF')

    #creating a dropbox for the user's pdf
    pdf = st.file_uploader("Upload your Pdf", type = 'pdf')

    #making sure that pdf is actually uploaded in order to execute the read()
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        #extracting pdf content
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        #splitting the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                                       chunk_overlap = 200, 
                                                       length_function = len)

        chunks = text_splitter(text)

 #we dont't want to vectorize a particular pdf more than once, as it may increase our api fee, thus we
 # store a vectorized pdf and if that particular pdf is uploaded again we use its pre-stored vectorization        
        store_name = pdf.name[:-4]
        
        #embedding the chunks
        if os.path.exists(f'{store_name}.pkl'):
            with open(f'{store_name}.pkl', 'rb') as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore =FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{store_name}', 'wb') as f:
                pickle.dump(VectorStore, f)
        
        #gettinf user query
        query = st.text_input("Ask Questionns About your PDF file:")
        st.write(query)

        #exectuing query relevant search, while making sure the context window meets the limit, thus choosing
        # a smaller value for 'k"
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            #choosing the llm model
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type='stuff')

            #the openai_callback ;ets us know the usage cost of the api withevery query
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)


if __name__ == "__main__":
    main()