#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
import boto3
import boto3.session
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM

    


#5c. Wrap within a function
def hr_index():
    #2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
    data_load=PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')

    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)

    #4. Create Embeddings -- Client connection
    
    # Configurando a sessão com o perfil SSO configurado
    session = boto3.Session(profile_name="ybadmin", region_name="us-east-1")
    bedrock_client = session.client("bedrock-runtime")

    # Criando embeddings do LangChain com o modelo Amazon Titan
       
    data_embeddings=BedrockEmbeddings(
        credentials_profile_name='ybadmin',
        model_id='amazon.titan-embed-text-v1'
    )

    #5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)


    #5b  Create index for HR Report
    db_index=data_index.from_loaders([data_load])
    
    return db_index
#6a. Write a function to connect to Bedrock Foundation Model
def hr_llm():
    llm= BedrockLLM(
        credentials_profile_name='ybadmin',
        model_id='meta.llama3-70b-instruct-v1:0',
        model_kwargs={
            "max_gen_len": 512,
            "temperature": 0.5 ,
            "top_p": 0.9})
    return llm

#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def hr_rag_response(index, question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question, llm=rag_llm)
    return hr_rag_query


# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html