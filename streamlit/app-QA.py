from requests.auth import HTTPBasicAuth
#from cohere_sagemaker import Client
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from utils import bedrock
import streamlit as st
import requests
import boto3
import json
import yaml
import re


MAX_LENGTH = 1024
NUM_RETURN_SEQUENCES = 1
TOP_K = 100
TOP_P = 0.9
DO_SAMPLE = True 
CONTENT_TYPE = 'application/json'

sagemaker_client = boto3.client('runtime.sagemaker')
#cohere_client = Client(endpoint_name=TEXT_GENERATION_ENDPOINT_NAME)

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
    endpoint_url=b_endpoint,
    region=b_region,
)

titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock)
br_embeddings = BedrockEmbeddings(client=boto3_bedrock)

# vector store index
docsearch = OpenSearchVectorSearch(
            opensearch_url=domain_endpoint,
            is_aoss=False,
            verify_certs = True,
            http_auth=(es_username, es_password),
            index_name = domain_index,
            embedding_function=br_embeddings)

memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
chat_history=[]

# --------------------------------- STREAMLIT APP --------------------------------- 

st.subheader('RFP Question & Answering')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Receive user input from the chat UI
prompt = st.text_input('Question: ', placeholder='Ask me anything ...', key='input')

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

qa = ConversationalRetrievalChain.from_llm(
    llm=titan_llm, 
    #retriever=vectorstore_faiss_aws.as_retriever(), 
    #retriever=vectorstore_faiss_aws.as_retriever(),
    retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 8, "vector_field":"embedding",  "text_field":    "passage", "metadata_field": "*"}),
    memory=memory_chain,
    #verbose=True,
    #condense_question_prompt=CONDENSE_QUESTION_PROMPT, # create_prompt_template(), 
    chain_type='stuff', # 'refine',
    #max_tokens_limit=100
)

qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
{context}

Answer the question inside the <q></q> XML tags. 

<q>{question}</q>

Do not use any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."

Answer:""")


def clean_text(text):
    # Use regular expression to match and remove any trailing characters after the last period.
    cleaned_text = re.sub(r'\.[^\.]*$', '.', text)
    return cleaned_text


if st.button('Submit', type='primary'):
    st.markdown('----')
    
    res_box = st.empty()
    
    if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
        result = "Thank you , that was a nice chat !!"
    elif len(prompt) > 0:
        try:
            result = qa.run({'question': prompt })
        except:
            result = "No answer"
            print(result)
           
    answer = result
    
    if len(answer) > 0:
        res_box.write(f'**Answer:**\n*{answer}*', unsafe_allow_html=False)

    res_box = st.empty()
    #res_box.markdown(f'**Reference**:\n*Document = {doc_id} | Passage = {passage_id} | Score = {score}*')
    res_box = st.empty()
    st.markdown('----')