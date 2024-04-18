import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

def get_pdf_text():
    pdf_path = <Mention your PDF path here>
    loader = PyPDFLoader(file_path =pdf_path)
    doc = loader.load()
    return doc

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap= 500,
    separators=["\n\n","\n"," ",""])
    text = text_splitter.split_documents(documents= text)
    return text

def get_vectorstore(text_chunks, query):
    embeddings = CohereEmbeddings(
        model="multilingual-22-12"
    )
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    #Prompttemplate

    prompt_template = """Text: {context}
    Question: {question}
    you are a chatbot designed to assist the users.
    Answer only the questions based on the text provided. If the text doesn't contain the answer,
    reply that the answer is not available.
    keep the answers precise to the question"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = { "prompt" : PROMPT }

    #LLM

    llm=Cohere(model="command", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs=chain_type_kwargs
    )

    response  = conversation_chain({"question": query})
    return(response.get("answer"))

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Assistant")

    # get pdf text
    raw_text = get_pdf_text()

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    user_question =  st.chat_input("Ask a Question")

    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = [{"role": "assistant",
                                         "content": "Hello there, how can i help you"}]

    if "messages" in st.session_state.keys():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if user_question is not None:
        st.session_state.messages.append({
            "role":"user",
            "content":user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading"):
                output = get_vectorstore(text_chunks, user_question)
                ai_response = output
                st.write(ai_response)

        new_ai_message = {"role":"assistant","content": ai_response}
        st.session_state.messages.append(new_ai_message)

if __name__ == '__main__':
    main()