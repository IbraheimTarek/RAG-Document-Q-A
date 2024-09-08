import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from docx import Document
import time
groq_api_key= os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key = groq_api_key, model = 'Llama3-8b-8192')

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know.
context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



if 'store' not in st.session_state:
    st.session_state.store = {}
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


### streamlit ui ###

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = st.chat_input("Enter Your Papers")

with st.sidebar:  # Alternatively, you can use st.container() for a specific area on the page
    if st.button("Document Embedding"):
        create_vector_embedding()
        st.write("Vector Database is ready")
    else:
        st.write('Click the button first to initialize the chat vector store')

if user_prompt :
    retriever = st.session_state.vectors.as_retriever()
    question_answer_chain  = create_stuff_documents_chain(llm,qa_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    rag_chain  = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": user_prompt},
        config={
            "configurable": {"session_id": "abc123"}
        }, 
    )

    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

    model_response = response['answer']
    # st.write(model_response)

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)


    def chunk_response(text, chunk_size=20):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    # Display assistant message with streaming effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder to update the message content
        full_response = ""
        for chunk in chunk_response(model_response):
            full_response += chunk
            message_placeholder.markdown(full_response)
            time.sleep(0.1)  # Simulate a delay between chunks to mimic streaming
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("----------------------------")


    st.session_state.messages.append({"role": "assistant", "content": model_response})