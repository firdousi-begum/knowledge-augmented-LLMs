import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate
from utils import bedrock
import streamlit as st
import requests
import boto3
import json
import yaml
import re


st.set_page_config(page_title="Chat with RFP Documents", page_icon="🦜")
st.title("🦜 Chat with RFP Documents")

sagemaker_client = boto3.client('runtime.sagemaker',
                                 aws_access_key_id='',
                                 aws_secret_access_key = '')

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

es_username = config['credentials']['username']
es_password = config['credentials']['password']

domain_endpoint = config['domain']['endpoint']
domain_index = config['domain']['index']

b_endpoint = config['bedrock-preview']['endpoint']
b_region = config['bedrock-preview']['region']

URL = f'{domain_endpoint}/{domain_index}/_search'

boto3_bedrock = bedrock.get_bedrock_client(
    aws_access_key_id='',
    aws_secret_access_key = '',
    endpoint_url=b_endpoint,
    region=b_region,
)

#@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    #for file in uploaded_files:
    #    temp_filepath = os.path.join(temp_dir.name, file.name)
    #    with open(temp_filepath, "wb") as f:
     #       f.write(file.getvalue())
     #   loader = PyPDFLoader(temp_filepath)
     #   docs.extend(loader.load())

    # Split documents
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    #splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    br_embeddings = BedrockEmbeddings(client=boto3_bedrock)
    
    # vector store index
    vectordb = OpenSearchVectorSearch(
            opensearch_url=domain_endpoint,
            is_aoss=False,
            verify_certs = True,
            http_auth=(es_username, es_password),
            index_name = domain_index,
            embedding_function=br_embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={"k": 8, "vector_field":"embedding",  "text_field":    "passage", "metadata_field": "*"})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, question, **kwargs):
        self.container.write(f"**Question:** {question}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["doc_name"])
            page = os.path.basename(str(doc.metadata["page"]))
            self.container.write(f"**Page {page} from {source}**")
            self.container.write(doc.page_content)



uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
#if not uploaded_files:
#    st.info("Please upload PDF documents to continue.")
 #   st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history",output_key='answer', chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
#llm = ChatOpenAI(
#    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
#)

titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock, model_kwargs = {"maxTokenCount":4096})
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=titan_llm, 
    #retriever=vectorstore_faiss_aws.as_retriever(), 
    #retriever=vectorstore_faiss_aws.as_retriever(),
    retriever =retriever,
    memory=memory,
    #verbose=True,
    #condense_question_prompt=CONDENSE_QUESTION_PROMPT, # create_prompt_template(), 
    chain_type='stuff', # 'refine',
    return_source_documents=True,
    max_tokens_limit=4096
)

qa_chain.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
{context}

Answer the question inside the <q></q> XML tags. 

<q>{question}</q>

Do not use any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."

Answer:""")

#qa_chain = ConversationalRetrievalChain.from_llm(
#    llm, retriever=retriever, memory=memory, verbose=True
#)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])
        st.write(response['answer'],unsafe_allow_html=True)
